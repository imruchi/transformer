import torch.nn as nn
from ffn import ffn
from mha import mha
import torch
class transformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048):
        super(transformerBlock, self).__init__()

        self.attention = mha(d_model, num_heads)
        self.ffn = ffn(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        output, attn_weights = self.attention(x,x,x)
        x = self.norm1(x + self.dropout(output))
        ffn = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn))
        return x
    

class encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, max_seq_len, vocab_size, d_ff):
        super(encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.posencoding = self.get_pos_encoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([transformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
    
    def get_pos_encoding(self, max_seq_len, d_model):
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        i = torch.arange(d_model).unsqueeze(0).float()

        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        return pos_enc.unsqueeze(0)  # Add batch dimension
    
    def forward(self, x):
        # Embedding + Positional Encoding
        x = self.embedding(x) + self.position_encoding[:, :x.size(1), :].to(x.device)
        
        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderBlock, self).__init__()
        self.attn1 = MultiHeadAttention(d_model, num_heads)  # Self-attention
        self.attn2 = MultiHeadAttention(d_model, num_heads)  # Encoder-decoder attention
        self.ffn = FeedForward(d_model, d_ff)  # Feed-forward network
        self.norm1 = nn.LayerNorm(d_model)  # Layer normalization for self-attention output
        self.norm2 = nn.LayerNorm(d_model)  # Layer normalization for encoder-decoder attention output
        self.norm3 = nn.LayerNorm(d_model)  # Layer normalization for feed-forward network output
        self.dropout = nn.Dropout(0.1)  # Dropout for regularization




        


    