import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # print("pe: ", self.pe.size())
        if x.dim()==3:
            return x + self.pe[:, :x.size(1)]
        else:
            return x + self.pe[:, :x.size(0)]
        
class NodeEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(NodeEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(1))
        
    def forward(self, x):
        return x + self.pe
    
    
    

# class ValueEmbedding(nn.Module):
#     def __init__(self, d_model):
#         """
#         Args:
#             d_model (int): The total size of the desired embedding vector.
#                             Half of this size is used for the value embedding.
#         """
#         super(ValueEmbedding, self).__init__()

#         # The MLP for embedding the value
#         self.value_mlp = nn.Sequential(
#             nn.Linear(1, d_model),
#             nn.LeakyReLU(negative_slope=0.01) # nn.GELU() or nn.ReLU()  #introduce some non-linearity
#             # add more layers if needed
#         )

#     def forward(self, x):
#         """
#         Forward pass of the ValueEmbedding.

#         Args:
#             x (torch.Tensor): The data samples to embed.
        
#         Returns:
#             torch.Tensor: The concatenated ValueEmbedding.
#         """
#         # Get the value embedding from the MLP
#         value_embedding = self.value_mlp(x)

#         return value_embedding

# class Embedding(nn.Module):
#     def __init__(self, d_model, max_seq_length):
#         super(Embedding, self).__init__()
#         self.d_model = d_model
#         self.value_embedding = ValueEmbedding(d_model)
#         self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
#     def forward(self, x):
#         """
#         Forward pass of the Embedding.

#         Args:
#             x (torch.Tensor): The data samples to embed, shape (N+1, S+1).
        
#         Returns:
#             torch.Tensor: The combined embeddings, shape (N+1, S+1, d_model).
#         """
#         N, S = x.size() #N = N+1, S = S+1
#         input = x[:, :-1] #(N+1)xS
#         input = input.reshape(-1,1) #flatten (N+1)S x 1

#         # Apply the value MLP to all columns except the last column
#         value_embedding = self.value_embedding(input)
#         value_embedding = value_embedding.view(N, S-1, -1)
        
#         # Zero vector for the value part in the last column
#         zero_value_embedding = torch.zeros(N, 1, self.d_model, device=x.device)
        
#         # Concatenate the value embeddings with the zero vector for the last column
#         value_embedding = torch.cat([value_embedding, zero_value_embedding], dim=1)
        
#         #Add positional encoding
#         embedding = self.positional_encoding(value_embedding)
        
#         return embedding