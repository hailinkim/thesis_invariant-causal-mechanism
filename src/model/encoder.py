import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append('/home/haikim20/angelicathesis')
# from src.model.embedding import Embedding

class SummaryWithAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SummaryWithAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads #Dimension of each head's key, query, and value (embedding dimension split to each head)
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model) #query transformation
        self.W_k = nn.Linear(d_model, d_model) #key transformation
        self.W_v = nn.Linear(d_model, d_model) #value transformation
        self.W_o = nn.Linear(d_model, d_model) #output transformation
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Transpose for dot product
        K = K.transpose(-2, -3)  
        # Apply scaled dot-product attention for each head
        scores = torch.einsum('npqk,nqsk->nqsp', [Q, K])/ math.sqrt(self.head_dim) #q: num heads, p:1, k:head dimension
        # Apply softmax to obtain weights for each value
        attention_weights = F.softmax(scores, dim=-1)
        # Compute the weighted sum of values across S samples
        weighted_values = torch.einsum('nqsp,nqsk->nqpk', [attention_weights, V.transpose(-2, -3)])
        return weighted_values
        
    def combine_heads(self, x): 
        concatenated = x.transpose(-2, -3).contiguous()
        return concatenated.view(-1, 1, self.d_model) # N x 1 x d_model

    def forward(self, Q, K, V, mask=None):
        _, S, _ = K.size()

        # Apply linear transformations and split heads
        Q = self.W_q(Q).view(-1, 1, self.n_heads, self.head_dim)
        K = self.W_k(K).view(-1, S, self.n_heads, self.head_dim)
        V = self.W_v(V).view(-1, S, self.n_heads, self.head_dim)
        # print("Q split: ", Q.size())
        # print("K split: ", K.size())
        # print("V split: ", V.size())
        
        # Perform scaled dot-product attention
        summary = self.scaled_dot_product_attention(Q, K, V, mask)
        # print("attn_output: ", summary.size())
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(summary))
        # print("output combined: ", output.size())
        return output


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, n_heads_summary, dropout):
        super(Encoder, self).__init__()
        # self.embedding = Embedding(d_model, n_steps)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model*4,  # Usually, the feedforward dimension is 4*d_model
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, 
            num_layers=n_layers
        )
        self.summary_attn = SummaryWithAttention(d_model, n_heads_summary)

    def forward(self, x): #x is embedded data
        output = self.transformer_encoder(x) # (N + 1) × (S + 1) lattice of representations        
        # Form queries using embeddings in column S+1
        Q = output[:-1, -1, :].unsqueeze(1) # N x 1 x d_model
        # Form keys and values using embeddings in column 1,..,S
        K = output[:-1, :-1, :] # N x S x d_model
        V = output[:-1, :-1, :] # N x S x d_model
        summary = self.summary_attn(Q, K, V) #N x 1 x d_model        
        return summary