import torch
import torch.nn as nn
import torch.nn.functional as F

class CaserNetwork(nn.Module):
    def __init__(self, item_num, hidden_size, state_size, num_filters, filter_sizes, dropout_rate):
        super(CaserNetwork, self).__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout_rate = dropout_rate

        # Embeddings
        self.state_embeddings = nn.Embedding(self.item_num + 1, self.hidden_size)

        # Horizontal convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, hidden_size)) for fs in filter_sizes
        ])

        # Vertical convolutional layer
        self.vertical_conv = nn.Conv2d(1, 1, (self.state_size, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters * len(filter_sizes) + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.output1 = nn.Linear(hidden_size, item_num)  # Q-values
        self.output2 = nn.Linear(hidden_size, item_num)  # CE logits

    def forward(self, inputs, is_training=True):
        # Embedding layer
        input_emb = self.state_embeddings(inputs)
        
        # Add channel dimension for convolution
        x = input_emb.unsqueeze(1)
        
        # Horizontal convolutions
        pooled_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x).squeeze(3))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            pooled_outputs.append(pooled)
        
        h_pool = torch.cat(pooled_outputs, 1)
        
        # Vertical convolution
        v_conv = F.relu(self.vertical_conv(x).squeeze(3))
        v_pool = v_conv.squeeze(2)
        
        # Concatenate horizontal and vertical features
        final = torch.cat([h_pool, v_pool], 1)
        
        # Fully connected layers with dropout
        fc1_out = F.relu(self.fc1(final))
        if is_training:
            fc1_out = self.dropout(fc1_out)
        
        # Output layers
        q_values = self.output1(fc1_out)
        ce_logits = self.output2(fc1_out)
        
        return q_values, ce_logits

    def get_embeddings(self):
        return self.state_embeddings.weight.data.cpu().numpy()