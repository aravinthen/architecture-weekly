# Program:     transformer.py
# Author:      Aravinthen Rajkumar
# Description: A reimplementation of the transformer from Vaswani et al. 2017

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    """
    Implementation of scaled dot product attention.
    """
    def __init__(self, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
        """
        Calculates the Attention for an input set of tensors.
        """
        # the embedding dimension
        dk = K.size(-1)
        
        # the scaled scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(dk)

        if mask is not None:
            # apply mask as defined in the paper.
            scores = scores.masked_fill(mask == 0, -1e9)

        # calculate the attention score
        attention_weights = F.softmax(scores, dim=-1)

        # apply dropout
        attention_weights = self.dropout(attention_weights)
        attention = torch.matmul(attention_weights, V)

        return attention
        
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int,  num_heads: int = 3):
        super().__init__()

        self.num_heads = num_heads

        # this is what allows the efficient parallelization of attention
        # you don't calculate each attenion head in sequence!
        self.head_dim = embed_dim // num_heads

        # attention module
        self.attention = Attention()
        
        self.Q_projection = nn.Linear(embed_dim, embed_dim)
        self.K_projection = nn.Linear(embed_dim, embed_dim)
        self.V_projection = nn.Linear(embed_dim, embed_dim)

        # took me ages to realise that all I had to do was this for the
        # concatenated vector!
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
        """
        Carries out a forward pass of batched multi-head attention.
        """
        batch_size, seq_len, embed_dim = Q.size()

        Q = self.Q_projection(Q)
        K = self.K_projection(K)
        V = self.V_projection(V)

        # split the outputs into heads
        Q = Q.view(batch_size,
                   seq_len,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)
        
        K = K.view(batch_size,
                   seq_len,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)
        
        V = V.view(batch_size,
                   seq_len,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)


        # context calculation for each head, as wel as recombination
        # concatenation is actually suboptimal: the goal here is to create as
        # few individual tensors as possible
        context = self.attention(Q, K, V, mask)

        # just using view was throwing up run-time errors
        context = context.transpose(1,2).contiguous().view(batch_size,
                                                           seq_len,
                                                           embed_dim)

        output = self.out_proj(context)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_length=1000):
        super().__init__()

        # follows the formula in section 3.5
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_terms)
        pe[:, 1::2] = torch.cos(position * div_terms)
        pe = pe.unsqueeze(0)

        # AI correction
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        # this bit required an AI correction. Need to figure out what is wrong...
        x = x + self.pe[:, :x.size(1), :]
        return x
        
        
class PositionFeedForwardNetwork(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, dropout:float = 0.1):
        """
        The position-wise feedforward network described in section 3.3
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim),
        )
        
    def forward(self, x):
        return self.layers(x)
        
    
class Encoder(nn.Module):
    def __init__(self, embed_dims: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__():

        self.multihead_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feedforward = PositoinFeedForwardNetwork(embed_dim, hidden_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Follows the basic implementation of the encoder diagram in the paper
        """
        # Q, K and V are all the same tensor in the encoder.
        attention = self.multihead_attention(x, x, x, mask)

        # residual connections + layer norm, followed by the positional feed
        # forward encoding ad another residual connection.
        x = self.norm1(x + self.dropout(attention))
        feed_forward_out = self.feedforward(x)
        x = self.norm2(x + self.dropout(feed_forward_out))

        return x
        
class Decoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = PositionFeedForwardNetwork(embed_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout

    def forward(self, x, encoder_out, source_mask, target_mask):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, encoder_out, encoder_out, source_mask)))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x
        

class Transformer(nn.Module):
    """
    This is a transformer written from scratch, intended to be as close to the 
    model demonstrated in Attention is All You Need.
    """
    def __init__(self,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 hidden_dim: int,
                 dropout: float,
                 max_sequence_length: int):
        
        super().__init__()

        # treating inputs
        self.source_embedding = nn.Embedding(source_vocab_size,
                                             embed_dim)
        self.target_embedding = nn.Embedding(source_vocab_size,
                                             embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim,
                                                      max_sequence_length)

        # encoder and decoder layers
        self.EncoderLayers = nn.ModuleList([
            Encoder(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.DecoderLayers = nn.ModuleList([
            Decoder(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.final_layer = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source, target):
        # Carry out encoding
        enc_x = self.dropout(self.positional_encoding(self.source_embedding(source)))
        for layer in self.EncoderLayers:
            enc_x = layer(enc_x)
            
        # Carry out decoding
        dec_x = self.dropout(self.positional_encoding(self.target_embedding(target)))
        for layer in self.decoder_layers:
            dec_x = layer(dec_x, enc_x)
            
        return self.final_layer(dec_x)
