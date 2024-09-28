import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from evaluators.evaluator import Evaluator


class Trainer:
    def __init__(self, model, target_model, mdp_params):
        self.model = model
        self.target_model = target_model
        self.mdp_params = mdp_params
        self.discount_factor = mdp_params['discount_factor']
        self.neg_samples = mdp_params['neg_samples']
        self.reward_buy = mdp_params['reward_buy']
        self.reward_click = mdp_params['reward_click']
        self.item_num = mdp_params['item_num']
        self.state_size = mdp_params['state_size']

    def compute_loss(self, model, target_model, batch, loss_type):
        state = torch.LongTensor(np.array(batch['state'].tolist()))
        len_state = torch.LongTensor(batch['len_state'].values)
        action = torch.LongTensor(batch['action'].values)
        next_state = torch.LongTensor(np.array(batch['next_state'].tolist()))  
        len_next_state = torch.LongTensor(batch['len_next_states'].values)
        is_done = torch.BoolTensor(batch['is_done'].values)
        is_buy = list(batch['is_buy'].values)

        reward = []
        for k in range(len(is_buy)):
            reward.append(self.reward_buy if is_buy[k] == 1 else self.reward_click)

        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        discount_tensor = torch.tensor([self.discount_factor] * len(action), dtype=torch.float32)

        reward_tensor = reward_tensor.to(state.device)
        discount_tensor = discount_tensor.to(state.device)

        q_values, ce_logits = model(state, len_state)

        with torch.no_grad():
            next_q_values_model, _ = model(next_state, len_next_state)
            next_q_values_target, _ = target_model(next_state, len_next_state)

            next_actions = next_q_values_model.argmax(dim=1)

            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze()

            next_q_values[is_done] = 0.0

        target_q_values = reward_tensor + self.discount_factor * next_q_values

        pos_q_values = q_values.gather(1, action.unsqueeze(1)).squeeze()
        q_loss = nn.MSELoss()(pos_q_values, target_q_values)

        ce_loss = nn.CrossEntropyLoss()(ce_logits, action)

        neg_q_loss = 0
        for _ in range(self.neg_samples):
            neg_action = torch.randint(0, self.item_num, action.shape)
            neg_q_values = q_values.gather(1, neg_action.unsqueeze(1)).squeeze()
            neg_q_loss += nn.MSELoss()(neg_q_values, torch.zeros_like(reward_tensor))
    
        if loss_type=='sa2c':
            avg_q = (pos_q_values+neg_q_values)/(1+self.neg_samples)
            advantage = pos_q_values-avg_q
            ce_loss_update = ce_loss * advantage 
            sa2c_loss = q_loss + neg_q_loss + ce_loss_update
            return torch.mean(sa2c_loss)
        
        elif loss_type=='snqn':
            total_loss = q_loss + ce_loss + neg_q_loss
            return total_loss
         
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}. Expected 'sa2c' or 'snqn'.")


    def train(self, replay_buffer, batch_size, num_epochs, optimizer, loss_type):
        num_rows = replay_buffer.shape[0]
        num_batches = int(num_rows / batch_size)
        total_step = 0
        
        for i in tqdm(range(num_epochs)):
            for j in tqdm(range(num_batches)):
                batch = replay_buffer.sample(n=batch_size)

                # Randomly decide which network to use for current and target
                if random.random() < 0.5:
                    current_model, eval_model = self.model, self.target_model
                else:
                    current_model, eval_model = self.target_model, self.model

                optimizer.zero_grad()

                if loss_type=='snqn':
                    loss = self.compute_loss(current_model, eval_model, batch, loss_type='snqn')
                elif loss_type=='sa2c':
                    if total_step < 15000:
                        loss = self.compute_loss(current_model, eval_model, batch, loss_type='snqn')
                    else:
                        loss = self.compute_loss(current_model, eval_model, batch, loss_type='sa2c')
                else:
                    raise ValueError(f"Invalid loss_type: {loss_type}. Expected 'sa2c' or 'snqn'.")

                loss.backward()
                optimizer.step()

                total_step += 1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))
                if total_step % 2000 == 0:
                    evaluator = Evaluator(
                        data_directory='recommender_system/data',
                        mdp_params=self.mdp_params,
                        topk=[5, 10, 20]
                    )
                    evaluator.evaluate(self.model, batch_size)