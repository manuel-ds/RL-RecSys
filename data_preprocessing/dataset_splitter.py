import os
import numpy as np
import pandas as pd
from utils.utils import to_pickled_df

class DatasetSplitter:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def split_dataset(self, sorted_events):
        total_sessions = sorted_events.session_id.unique()
        np.random.shuffle(total_sessions)
        
        fractions = np.array([0.8, 0.1, 0.1])
        train_ids, val_ids, test_ids = np.array_split(
            total_sessions, (fractions[:-1].cumsum() * len(total_sessions)).astype(int))
        
        train_sessions = sorted_events[sorted_events['session_id'].isin(train_ids)]
        val_sessions = sorted_events[sorted_events['session_id'].isin(val_ids)]
        test_sessions = sorted_events[sorted_events['session_id'].isin(test_ids)]
        
        to_pickled_df(self.data_directory, sampled_train=train_sessions)
        to_pickled_df(self.data_directory, sampled_val=val_sessions)
        to_pickled_df(self.data_directory, sampled_test=test_sessions)
        
        return train_sessions, val_sessions, test_sessions