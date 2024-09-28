import pandas as pd
import os
import torch
from utils.utils import pad_history, calculate_hit

class Evaluator:
    def __init__(self, data_directory, mdp_params, topk):
        self.data_directory = data_directory
        self.reward_buy = mdp_params['reward_buy']
        self.reward_click = mdp_params['reward_click']
        self.item_num = mdp_params['item_num']
        self.state_size = mdp_params['state_size']
        self.topk = topk

    def evaluate(self, model, batch_size):
        eval_sessions = pd.read_pickle(os.path.join(self.data_directory, 'sampled_val.df'))
        eval_ids = eval_sessions.session_id.unique()
        groups = eval_sessions.groupby('session_id')
        
        evaluated = 0
        total_clicks = 0.0
        total_purchase = 0.0
        total_reward = [0] * len(self.topk)
        hit_clicks = [0] * len(self.topk)
        ndcg_clicks = [0] * len(self.topk)
        hit_purchase = [0] * len(self.topk)
        ndcg_purchase = [0] * len(self.topk)

        model.eval()

        while evaluated < len(eval_ids):
            states, len_states, actions, rewards = [], [], [], []

            for i in range(batch_size):
                if evaluated == len(eval_ids):
                    break

                session_id = eval_ids[evaluated]
                group = groups.get_group(session_id)
                history = []

                for index, row in group.iterrows():
                    state = list(history)
                    len_states.append(self.state_size if len(state) >= self.state_size else 1 if len(state) == 0 else len(state))
                    state = pad_history(state, self.state_size, self.item_num)
                    states.append(state)

                    action = row['item_id']
                    is_buy = row['is_buy']
                    reward = self.reward_buy if is_buy == 1 else self.reward_click

                    if is_buy == 1:
                        total_purchase += 1.0
                    else:
                        total_clicks += 1.0

                    actions.append(action)
                    rewards.append(reward)
                    history.append(row['item_id'])

                evaluated += 1

            states_tensor = torch.LongTensor(states).to(model.state_embeddings.weight.device)
            len_states_tensor = torch.LongTensor(len_states).to(model.state_embeddings.weight.device)

            with torch.no_grad():
                _, ce_logits = model(states_tensor, len_states_tensor)
                sorted_indices = torch.argsort(ce_logits, dim=1, descending=True).cpu().numpy()

            calculate_hit(sorted_indices, self.topk, actions, rewards, self.reward_click, total_reward, hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase)

        # Print evaluation metrics
        print('#############################################################')
        print(f'total clicks: {int(total_clicks)}, total purchase: {int(total_purchase)}')
        for i in range(len(self.topk)):
            hr_click = hit_clicks[i] / total_clicks if total_clicks > 0 else 0
            hr_purchase = hit_purchase[i] / total_purchase if total_purchase > 0 else 0
            ng_click = ndcg_clicks[i] / total_clicks if total_clicks > 0 else 0
            ng_purchase = ndcg_purchase[i] / total_purchase if total_purchase > 0 else 0

            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(f'cumulative reward @ {self.topk[i]}: {total_reward[i]:.6f}')
            print(f'clicks HR NDCG @ {self.topk[i]}: {hr_click:.6f}, {ng_click:.6f}')
            print(f'purchase HR NDCG @ {self.topk[i]}: {hr_purchase:.6f}, {ng_purchase:.6f}')
        print('#############################################################')
        
        model.train()