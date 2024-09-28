import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SASRecBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super(SASRecBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SASRecNetwork(nn.Module):
    def __init__(self, item_num, hidden_size, state_size, num_blocks, num_heads, dropout_rate):
        super(SASRecNetwork, self).__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_blocks = num_blocks
        
        self.item_embeddings = nn.Embedding(self.item_num + 1, self.hidden_size)
        self.positional_encoding = PositionalEncoding(self.hidden_size, max_len=state_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_size, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
        
        self.output1 = nn.Linear(hidden_size, item_num)  # Q-values
        self.output2 = nn.Linear(hidden_size, item_num)  # CE logits
        
    def forward(self, inputs, len_state, is_training=True):
        mask = (inputs != self.item_num).float().unsqueeze(-1)
        
        # Item embedding and positional encoding
        seq = self.item_embeddings(inputs)
        seq = self.positional_encoding(seq)
        
        # Apply mask and dropout
        seq = seq * mask
        seq = self.dropout(seq) if is_training else seq
        
        # Create attention mask (lower triangular matrix)
        attn_mask = torch.tril(torch.ones(inputs.size(1), inputs.size(1))).bool().to(inputs.device)
        
        # Apply SASRec blocks
        for block in self.blocks:
            seq = block(seq.transpose(0, 1), attn_mask).transpose(0, 1)
            seq = seq * mask
        
        # Extract the hidden state of the last element in the sequence
        self.states_hidden = seq[torch.arange(seq.size(0)), len_state - 1]
        
        # Output layers
        q_values = self.output1(self.states_hidden)
        ce_logits = self.output2(self.states_hidden)
        
        return q_values, ce_logits
    
    def get_embeddings(self):
        return self.item_embeddings.weight.data.cpu().numpy()