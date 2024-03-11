import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, d_model, n_nodes, hidden_size, n_layers, n_heads, dropout, batch_first):
        super(Decoder, self).__init__()

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model*4,  # Usually, the feedforward dimension is 4*d_model
            dropout=dropout,
            batch_first = batch_first
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, 
            num_layers=n_layers
        )
        # self.fc = nn.Linear(d_model, 1)
        # self.dropout = nn.Dropout(dropout)
        # self.sigmoid = nn.Sigmoid()

        self.auxiliary_mlp = AuxiliaryMLP(d_model, hidden_size, n_nodes)

    def forward(self, tgt_embed, encoder_summary, tgt_mask): 
        # print("tgt embed: ", tgt_embed.size())
        output = self.transformer_decoder(tgt = tgt_embed, memory = encoder_summary, tgt_mask = tgt_mask)
        # print("decoder output: ", output.size())
        # prediction = self.fc(output) # raw prediction (logit) for each element in the target sequence
        # print(prediction)
        # print(prediction.squeeze(2))
        # print("logit size: ", prediction.squeeze(2).size())
        parent, children = self.auxiliary_mlp(encoder_summary)
        return output, parent, children

class AuxiliaryMLP(nn.Module): #this MLP takes as input encoder summary of each node and predicts its parents and children
    def __init__(self, d_model, hidden_size, n_nodes):
        super(AuxiliaryMLP, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_size) #1st hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) #2nd hidden layer
        self.act = nn.LeakyReLU() #activation function
        # Linear layer for predicting parents for each node
        self.fc_parents = nn.Linear(hidden_size, n_nodes)
        # Linear layer for predicting children for each node
        self.fc_children = nn.Linear(hidden_size, n_nodes)

    def forward(self, x): #input x is the encoder summary
        # x = x.squeeze(0) #(N, d_model)
        # Predict parents and children separately
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        parents = self.fc_parents(x) #self.sigmoid(self.fc_parents(x))
        children = self.fc_children(x) #self.sigmoid(self.fc_children(x))
        return parents.view(1,-1), children.transpose(1,2).reshape(1,-1) #logits