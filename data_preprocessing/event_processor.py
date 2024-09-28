import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.utils import to_pickled_df

class EventProcessor:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def process_events(self):
        event_df = pd.read_csv(os.path.join(self.data_directory, 'events.csv'), header=0)
        event_df.columns = ['timestamp', 'session_id', 'behavior', 'item_id', 'transid']
        
        event_df = event_df[event_df['transid'].isnull()]
        event_df = event_df.drop('transid', axis=1)
        
        event_df['valid_session'] = event_df.session_id.map(event_df.groupby('session_id')['item_id'].size() > 2)
        event_df = event_df.loc[event_df.valid_session].drop('valid_session', axis=1)
        
        event_df['valid_item'] = event_df.item_id.map(event_df.groupby('item_id')['session_id'].size() > 2)
        event_df = event_df.loc[event_df.valid_item].drop('valid_item', axis=1)
        
        item_encoder = LabelEncoder()
        session_encoder = LabelEncoder()
        behavior_encoder = LabelEncoder()
        event_df['item_id'] = item_encoder.fit_transform(event_df.item_id)
        event_df['session_id'] = session_encoder.fit_transform(event_df.session_id)
        event_df['behavior'] = behavior_encoder.fit_transform(event_df.behavior)
        
        event_df['is_buy'] = 1 - event_df['behavior']
        event_df = event_df.drop('behavior', axis=1)
        
        sorted_events = event_df.sort_values(by=['session_id', 'timestamp'])
        sorted_events.to_csv('data/sorted_events.csv', index=None, header=True)
        to_pickled_df(self.data_directory, sorted_events=sorted_events)
        return sorted_events