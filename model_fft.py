import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
    
def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1))
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    return torch.softmax(m, -1)

def attention(Q, K, V):
    a = a_norm(Q, K)  # (batch_size, hidden_size, timestepgth)
    return torch.matmul(a, V)  # (batch_size, timestepgth, timestepgth)

class QKV(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(QKV, self).__init__()
        self.fc = nn.Linear(embedding_size, hidden_size, bias=False)
    def forward(self, x):
        return self.fc(x)
    

class AttentionBlock(nn.Module):
    def __init__(self, feat_dim, embed_dim):
        super(AttentionBlock, self).__init__()
        self.value = QKV(feat_dim, embed_dim)
        self.key = QKV(feat_dim, embed_dim)
        self.query = QKV(feat_dim, embed_dim)

    def forward(self, x, kv=None):
        if kv is None:
            qkv = attention(self.query(x), self.key(x), self.value(x))
            return qkv
        else:
            qkv = attention(self.query(x), self.key(kv), self.value(kv))
            return qkv

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, feat_dim, embed_dim, num_heads):
        '''input: embedding_size, output: embedding_size'''
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(num_heads):
            self.heads.append(AttentionBlock(feat_dim, embed_dim))
        self.heads = nn.Sequential(*self.heads)
        self.fc = nn.Linear(num_heads * embed_dim, feat_dim, bias=False)
        
    def forward(self, x, kv=None):
        a = []
        for h in self.heads:
            a.append(h(x, kv=kv))
        a = torch.stack(a, dim=-1)  # combine heads
        a = a.flatten(start_dim=2)  # flatten all head outputs
        x = self.fc(a)
        return x

class Aug_FFT(nn.Module):
    def __init__(self, feat_dim, embed_dim):
        super(Aug_FFT, self).__init__()
        self.value = QKV(feat_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, feat_dim, bias=False)
        
    def forward(self, x):
        x = self.value(x)
        x = self.fc(torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, feat_dim=188, num_heads=4, ff_dim=64, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttentionBlock(feat_dim, embed_dim, num_heads)
        self.fft = Aug_FFT(feat_dim, embed_dim)
        self.ffn = nn.Sequential(nn.Linear(feat_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, feat_dim))
        self.ln1 = nn.LayerNorm([13, feat_dim], eps=1e-6)
        self.ln2 = nn.LayerNorm([13, feat_dim], eps=1e-6)
        self.dp1 = nn.Dropout(rate)
        self.dp2 = nn.Dropout(rate)
        
    def forward(self, inputs):
        attn_output = self.att(inputs)  # inputs bs,13,188, output bs,13,188
        fft_output = self.fft(inputs)  # inputs bs,13,188, output bs,13,188
        attn_output = self.dp1(attn_output) 
        out1 = self.ln1(inputs+attn_output+fft_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dp2(ffn_output)
        return self.ln2(out1+ffn_output)

class Transformer(nn.Module):
    def __init__(self, num_tokens=1,
                        embed_dim=128,
                        feat_dim=188,
                        num_heads=4,
                        ff_dim=64,
                        num_encoder_layers=2,
                        dropout_p=0.3):
        super(Transformer, self).__init__()
        # LAYERS
        self.embeddings = {}
        for k in range(11):
            self.embeddings[k] = nn.Embedding(10, 4).to('cuda:0')
        feat_dim2 = feat_dim-11+11*4
        self.encoder = nn.Linear(feat_dim2, feat_dim) # bs,13,feat_dim

        # Initiate encoder and Decoder layers
        self.encs = []
        for i in range(num_encoder_layers):
            self.encs.append(TransformerBlock(embed_dim, feat_dim, num_heads, ff_dim, rate=0.1))
        self.encs = nn.Sequential(*self.encs)

#         # Dense layers for managing network inputs and outputs
#         self.enc_input_fc = nn.Linear(feature, embedding_size)
#         self.enc_dropout = nn.Dropout(dropout)
        
#         self.pos = PositionalEncoding(embedding_size)
        self.out_fc1 = nn.Linear(feat_dim, 64)
        self.out_fc2 = nn.Linear(64, 32)
        self.out_fc3 = nn.Linear(32, 1)

    def forward(self, src):
        # emb
        src_cat = []
        for k in range(11):
            src_cat.append(self.embeddings[k](src[:,:,k].long())) #* math.sqrt(self.hidden_size)
        src_cat = torch.cat(src_cat, -1)
        src = torch.cat([src_cat, src[:,:,11:]], -1)
        src = self.encoder(src)
        
        # encoder
        for enc in self.encs:
            src_old = src
            src = enc(src)
            src = 0.9*src + 0.1*src_old
        src = F.elu(self.out_fc1(src[:,-1,:]))
        src = F.elu(self.out_fc2(src))
        src = self.out_fc3(src)
        return src