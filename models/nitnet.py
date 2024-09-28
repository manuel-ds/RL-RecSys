import torch
import torch.nn as nn
import torch.nn.functional as F

class NextItNetResidualBlock(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size, causal=True):
        super(NextItNetResidualBlock, self).__init__()
        self.causal = causal
        self.dilation = dilation
        
        self.c1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.c2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.c3 = nn.Conv1d(in_channels, out_channels, 1)
        
    def forward(self, x):
        residual = x
        
        if self.causal:
            padding = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (padding, 0))
        
        out = self.c1(x)
        out = F.relu(out)
        out = self.c2(out)
        out = F.relu(out)
        
        residual = self.c3(residual)
        return out + residual

class NItNetNetwork(nn.Module):
    def __init__(self, item_num, hidden_size, state_size, dilated_channels=64, 
                 dilations=[1,2,1,2,1,2], kernel_size=3):
        super(NItNetNetwork, self).__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.model_para = {
            'dilated_channels': dilated_channels,
            'dilations': dilations,
            'kernel_size': kernel_size,
        }
        
        self.state_embeddings = nn.Embedding(self.item_num + 1, self.hidden_size)
        
        self.residual_blocks = nn.ModuleList([
            NextItNetResidualBlock(
                dilation, 
                hidden_size if i == 0 else dilated_channels, 
                dilated_channels, 
                kernel_size,
                causal=True
            ) for i, dilation in enumerate(self.model_para['dilations'])
        ])
        
        self.final_layer = nn.Linear(dilated_channels, hidden_size)
        self.output1 = nn.Linear(hidden_size, item_num)  # Q-values
        self.output2 = nn.Linear(hidden_size, item_num)  # CE logits
        
    def forward(self, inputs, len_state, is_training=True):
        mask = (inputs != self.item_num).float().unsqueeze(-1)
        context_embedding = self.state_embeddings(inputs) * mask
        
        dilate_output = context_embedding.transpose(1, 2)  # Change to channel-first format
        for block in self.residual_blocks:
            dilate_output = block(dilate_output)
            dilate_output = dilate_output * mask.transpose(1, 2)
        
        dilate_output = dilate_output.transpose(1, 2)  # Change back to batch-first format
        
        self.states_hidden = dilate_output[torch.arange(dilate_output.size(0)), len_state - 1]
        
        output = self.final_layer(self.states_hidden)
        q_values = self.output1(output)
        ce_logits = self.output2(output)
        
        return q_values, ce_logits
    
    def get_embeddings(self):
        return self.state_embeddings.weight.data.cpu().numpy()