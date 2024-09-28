# recommender_system/utils/utils.py
import numpy as np
import os 

def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))


def pad_history(state, state_size, item_num):
    """Pads the history to the maximum state size."""
    if len(state) < state_size:
        state = state + [item_num] * (state_size - len(state))
    return state[-state_size:]

def calculate_hit(sorted_indices, topk, actions, rewards, reward_click, total_reward, hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase):
    """Calculates hit rate and NDCG for clicks and purchases."""
    for k in range(len(topk)):
        for i in range(len(actions)):
            action = actions[i]
            reward = rewards[i]
            
            if reward == reward_click:
                if action in sorted_indices[i][:topk[k]]:
                    hit_clicks[k] += 1
                    ndcg_clicks[k] += 1.0 / np.log2(np.where(sorted_indices[i] == action)[0][0] + 2)
            else:
                if action in sorted_indices[i][:topk[k]]:
                    hit_purchase[k] += 1
                    ndcg_purchase[k] += 1.0 / np.log2(np.where(sorted_indices[i] == action)[0][0] + 2)

            total_reward[k] += reward