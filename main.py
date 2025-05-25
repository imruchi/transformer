import torch.nn as nn
from ffn import ffn
from mha import mha

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
        



        


    