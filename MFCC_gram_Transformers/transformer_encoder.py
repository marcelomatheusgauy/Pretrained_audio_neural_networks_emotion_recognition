import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import sklearn.metrics
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import os
import pandas

    
#we only need the encoder part of the transformer
class TransformerEncoder(nn.Module):
    def __init__(self, encoder, src_embed, generator, ri_generator, gen_generator, age_generator, age_2_generator):
        super(TransformerEncoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator
        self.ri_generator = ri_generator
        self.gen_generator = gen_generator
        self.age_generator = age_generator
        self.age_2_generator = age_2_generator
        
    def forward(self, src, src_mask):
        return self.encode(src, src_mask)
    
    def encode(self, src, src_mask):
        #print('src shape', src.shape)
        #print('srcembed shape', self.src_embed(src).shape)
        #print('src_mask shape', src_mask.shape)
        return self.encoder(self.src_embed(src), src_mask)
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab, pretrain='pretrain'):
        super(Generator, self).__init__()
        self.proj1 = nn.Linear(d_model, d_model)
        self.layernorm = LayerNorm(d_model)
        self.proj2 = nn.Linear(d_model, vocab)
        self.pretrain = pretrain

    def forward(self, x):
        if self.pretrain == 'pretrain':
            #x = F.relu(self.proj1(x))
            #x = self.layernorm(x)
            x = self.proj2(x)
            return x
        elif self.pretrain == 'ri':
            #x = F.relu(self.proj1(x))
            #x = self.layernorm(x)
            x = F.softmax(self.proj2(x), dim=-1)
            return x
        elif self.pretrain == 'gen':
            x = F.softmax(self.proj2(x), dim=-1)
            return x
        #elif age
        #doing softmax here - use NLLloss below as crossentropyloss does softmax already
        #x = F.softmax(self.proj2(x), dim=-1)
        x = self.proj2(x)
        return x
    
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        #print('x', x.shape)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        #print('mask ', mask.shape)
        #print('scores ', scores.shape)
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
    


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


#we only need to use the encoder part of the transformers
def make_transformer_encoder_model(n_coeffs, ri_out_coeffs, gen_out_coeffs, age_out_coeffs, age_2_out_coeffs, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    #check nn.Linear operation
    model = TransformerEncoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(nn.Linear(n_coeffs, d_model), c(position)),
        Generator(d_model, n_coeffs, 'pretrain'),
        Generator(d_model, ri_out_coeffs, 'ri'),
        Generator(d_model, gen_out_coeffs, 'gen'),
        Generator(d_model, age_out_coeffs, 'age'),
        Generator(d_model, age_2_out_coeffs, 'age_2'))
    model.cuda()
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
