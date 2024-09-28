import os
import pandas as pd
from utils.utils import to_pickled_df, pad_history

class BufferCreator:
    def __init__(self, data_directory, history_length):
        self.data_directory = data_directory
        self.history_length = history_length

    def create_buffer(self, sorted_events):
        item_ids = sorted_events.item_id.unique()
        pad_item = len(item_ids)
        
        train_sessions = pd.read_pickle(os.path.join(self.data_directory, 'sampled_train.df'))
        groups = train_sessions.groupby('session_id')
        ids = train_sessions.session_id.unique()
        
        state, len_state, action, is_buy, next_state, len_next_state, is_done = [], [], [], [], [], [], []
        for id in ids:
            group = groups.get_group(id)
            history = []
            for index, row in group.iterrows():
                s = list(history)
                len_state.append(self.history_length if len(s) >= self.history_length else 1 if len(s) == 0 else len(s))
                s = pad_history(s, self.history_length, pad_item)
                a = row['item_id']
                is_b = row['is_buy']
                state.append(s)
                action.append(a)
                is_buy.append(is_b)
                history.append(row['item_id'])
                next_s = list(history)
                len_next_state.append(self.history_length if len(next_s) >= self.history_length else 1 if len(next_s) == 0 else len(next_s))
                next_s = pad_history(next_s, self.history_length, pad_item)
                next_state.append(next_s)
                is_done.append(False)
            is_done[-1] = True
        
        replay_buffer = pd.DataFrame({
            'state': state,
            'len_state': len_state,
            'action': action,
            'is_buy': is_buy,
            'next_state': next_state,
            'len_next_states': len_next_state,
            'is_done': is_done
        })
        
        to_pickled_df(self.data_directory, replay_buffer=replay_buffer)
        
        data_statistics = pd.DataFrame({
            'state_size': [self.history_length],
            'item_num': [pad_item]
        })
        to_pickled_df(self.data_directory, data_statistics=data_statistics)
        
        return replay_buffer, data_statistics