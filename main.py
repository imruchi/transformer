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
    

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, vocab_size, max_seq_len):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  
        self.position_encoding = self.get_position_encoding(max_seq_len, d_model) 
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x, enc_output):
        x = self.embedding(x) + self.position_encoding[:, :x.size(1), :].to(x.device)
        for layer in self.layers:
            x = layer(x, enc_output)
        return x



class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderBlock, self).__init__()
        self.attn1 = mha(d_model, num_heads)  # Self-attention
        self.attn2 = mha(d_model, num_heads)  # Encoder-decoder attention
        self.ffn = mha(d_model, d_ff)  # Feed-forward network
        self.norm1 = nn.LayerNorm(d_model)  # Layer normalization for self-attention output
        self.norm2 = nn.LayerNorm(d_model)  # Layer normalization for encoder-decoder attention output
        self.norm3 = nn.LayerNorm(d_model)  # Layer normalization for feed-forward network output
        self.dropout = nn.Dropout(0.1)  # Dropout for regularization


    def forward(self, x, enc_output):
        attn_output, _ = self.attn1(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        attn_output, _ = self.attn2(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x
    


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, vocab_size, max_seq_len):
        super(Transformer, self).__init__()
        self.encoder = encoder(d_model, num_heads, num_layers, d_ff, vocab_size, max_seq_len)
        self.decoder = DecoderBlock(d_model, num_heads, num_layers, d_ff, vocab_size, max_seq_len)
        self.output_layer = nn.Linear(d_model, vocab_size)  # Output layer for predicting vocab tokens

    def forward(self, input_seq, target_seq):
        enc_output = self.encoder(input_seq)  # Encoder processes the input sequence
        dec_output = self.decoder(target_seq, enc_output)  # Decoder processes the target sequence with encoder's output
        return self.output_layer(dec_output)  # Linear layer to produce output logits



# model = Transformer(d_model=512, num_heads=8, num_layers=6, d_ff=2048, vocab_size=10000, max_seq_len=512)

# # some dummy input data (e.g., tokenized sentences)
# input_seq = torch.randint(0, 10000, (2, 10))  
# target_seq = torch.randint(0, 10000, (2, 10))  

# # Forward pass through the model
# output = model(input_seq, target_seq)




        


    