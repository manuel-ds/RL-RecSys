import torch.nn as nn
import torch 

class GRUQNetwork(nn.Module):
    def __init__(self, item_num, hidden_size, state_size):
        super(GRUQNetwork, self).__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size
        self.state_size = state_size

        self.state_embeddings = nn.Embedding(self.item_num + 1, self.hidden_size)
        
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        
        self.output1 = nn.Linear(self.hidden_size, self.item_num)  # Q-values
        self.output2 = nn.Linear(self.hidden_size, self.item_num)  # CE logits

    def forward(self, inputs, len_state):
        input_emb = self.state_embeddings(inputs)

        packed_input = nn.utils.rnn.pack_padded_sequence(input_emb, len_state, batch_first=True, enforce_sorted=False)

        gru_out, self.hidden = self.gru(packed_input)
        
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        batch_size = inputs.size(0)
        gru_out = gru_out[torch.arange(batch_size), torch.tensor(len_state) - 1]

        q_values = self.output1(gru_out)
        ce_logits = self.output2(gru_out)

        return q_values, ce_logits

    def get_embeddings(self):
        return self.state_embeddings.weight.data.cpu().numpy()