from models.gru import GRUQNetwork
from trainers.trainer import Trainer
from evaluators.evaluator import Evaluator
from data_preprocessing.event_processor import EventProcessor
from data_preprocessing.dataset_splitter import DatasetSplitter
from data_preprocessing.buffer_creator import BufferCreator
import torch 
import yaml


if __name__ == '__main__':
    data_directory = 'data'
    
    event_processor = EventProcessor(data_directory)
    sorted_events = event_processor.process_events()
    
    dataset_splitter = DatasetSplitter(data_directory)
    train_sessions, val_sessions, test_sessions = dataset_splitter.split_dataset(sorted_events)
    
    buffer_creator = BufferCreator(data_directory, history_length=10)
    replay_buffer, data_statistics = buffer_creator.create_buffer(sorted_events)

    with open('mdp_params.yml', 'r') as file:
        mdp_params = yaml.safe_load(file)

    mdp_params['state_size']=data_statistics['state_size'][0]  
    mdp_params['item_num'] = data_statistics['item_num'][0] 

    with open('model_params.yml', 'r') as file:
        model_params = yaml.safe_load(file)

    # example using GRU Q-network
    model = GRUQNetwork(item_num=mdp_params['item_num'], hidden_size=model_params['hidden_size'], state_size=mdp_params['state_size'])
    target_model = GRUQNetwork(item_num=mdp_params['item_num'], hidden_size=model_params['hidden_size'], state_size=mdp_params['state_size'])
    
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(
        model=model,
        target_model=target_model,
        mdp_params=mdp_params)
    
    trainer.train(replay_buffer, model_params['batch_size'], model_params['num_epochs'], optimizer, loss_type='sa2c')
    
