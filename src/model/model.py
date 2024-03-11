import torch.nn as nn
import torch
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.embedding import PositionalEncoding, NodeEncoding
import sys
sys.path.append('/home/haikim20/angelicathesis')

class Transformer(nn.Module):
    def __init__(self, n_nodes, max_seq_length_src, d_model, n_layers, n_layers_dec, n_heads, n_heads_summary, hidden_size, dropout):
        super(Transformer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_nodes = n_nodes
        #source embedding
        self.src_embedding = nn.Conv1d(in_channels=n_nodes+1, 
                                      out_channels=n_nodes+1,
                                      kernel_size=3, #max_lag+1
                                      padding='same', padding_mode = 'replicate',
                                      groups = n_nodes+1) #Embedding(d_model, max_seq_length_src) # max_seq_length_src = n_samples + 1
        self.mlp_embedding = nn.Linear(1, d_model)
        #target embedding
        self.tgt_embedding = nn.Embedding(3, d_model, padding_idx=2) 
        self.src_pe = PositionalEncoding(d_model, max_seq_length_src)
        self.node_pe = NodeEncoding(d_model, n_nodes+1)
        self.tgt_pe = PositionalEncoding(d_model, n_nodes**2)
        self.enc = Encoder(d_model, n_layers, n_heads, n_heads_summary, dropout)
        self.dec = Decoder(d_model, n_nodes, hidden_size, n_layers_dec, n_heads, dropout, batch_first=True)
        self.generator = nn.Linear(d_model, 1) #transforms decoder output to logits
        self.sigmoid = nn.Sigmoid() #transforms logits to probabilities
        
    # def generate_tgt_mask(self, tgt):
    #     device = tgt.get_device()
    #     tgt_mask = (tgt != 2).unsqueeze(1).unsqueeze(3)
    #     seq_length = tgt.size(1)
    #     nopeek_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=device), diagonal=0)).bool()
    #     tgt_mask = tgt_mask & nopeek_mask
    #     return tgt_mask.squeeze(0)
    
    def generate_look_ahead_mask(self, tgt):
        device = tgt.get_device()
        tgt_seq_len = tgt.shape[1]
        mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device = device)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    

    def forward(self, src, tgt):
        """
        
        tgt: flattened adjacency matrix (1,sequence_length) 
        """
        conv = self.src_embedding(src[:,:,:-1]) #(1, N+1, S)
        # flatten the embedding for MLP by timesteps (every node for a timestep, and then the next timestep, and so on)
        conv_flatten = conv.transpose(1,2).reshape(-1,1) #((N+1)S, 1))
        
        # mlp = self.mlp_embedding(conv_flatten)#.\ #((N+1)S, d_model); embeddings for every node in a time step, and then the next timestep
        # mlp = mlp.reshape(-1, self.n_nodes+1, self.d_model)#.\ #(S, N+1, d_model)
        # mlp = mlp.transpose(1,0) #(N+1, S, d_model)
        
        mlp = self.mlp_embedding(conv_flatten).\
                reshape(-1, self.n_nodes+1, self.d_model).\
                transpose(1,0) #(N+1, S, d_model)
        #add zeros for value embedding for column S+1
        node_id_embed = torch.zeros(self.n_nodes+1, 1, self.d_model, device = src.device)
        value_embed = torch.cat([mlp, node_id_embed], dim=1)
        #src embedding
        src_embedded = self.node_pe(self.src_pe(value_embed)) #(N+1) x (S+1) x d_model
        
        #target embedding
        tgt_embedded = self.tgt_pe(self.tgt_embedding(tgt.squeeze(0)))#.to(device) #(1, N^2, d_model)
#         #target mask
        tgt_mask = self.generate_look_ahead_mask(tgt)#.to(device)

#         # encoder output (summary vector for each node)
        summary = self.enc(src_embedded) #(N, 1, d_model)
#         # reshape encoder output
        summary = summary.squeeze(1).unsqueeze(0) #(1, N, d_model)
        output, parent, children = self.dec(tgt_embedded, summary, tgt_mask) 

        return output, parent, children
    
    def encode(self, src):
        conv = self.src_embedding(src[:,:,:-1])
        conv_flatten = conv.transpose(1,2).reshape(-1,1)
        mlp = self.mlp_embedding(conv_flatten).\
                reshape(-1, self.n_nodes+1, self.d_model).\
                transpose(1,0)
        #add zeros for value embedding for column S+1
        node_id_embed = torch.zeros(self.n_nodes+1, 1, self.d_model, device = src.device)
        value_embed = torch.cat([mlp, node_id_embed], dim=1)
        #src embedding
        src_embedded = self.node_pe(self.src_pe(value_embed))
        # src_embedded = self.node_pe(self.src_pe(self.src_embedding(src[:,:,:-1])))
        summary = self.enc(src_embedded)
        return summary
    
    def decode(self, tgt, summary):
        tgt_embedded = self.tgt_pe(self.tgt_embedding(tgt.squeeze(0)))#.to(device) #(1, N^2, d_model)
        tgt_mask = self.generate_look_ahead_mask(tgt)#.to(device)
        output, parent, children = self.dec(tgt_embedded, summary, tgt_mask) 
        return output, parent, children