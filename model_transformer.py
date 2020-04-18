import torch
import numpy as np
import torch.nn as nn
from torch.nn import TransformerEncoder



class classifier(nn.Module):
    
    #define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):
        
        #Constructor
        super().__init__()          
        
        #embedding layer
        embedding_dim = 64
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.init_weights()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.fc = nn.Linear(embedding_dim, output_dim)
        # self.fc = nn.Linear(hidden_dim * 2, mid_dim)
        # self.fc2 = nn.Linear(mid_dim, output_dim)
        #activation function
        self.act = nn.Sigmoid()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, text, text_lengths):
        embedded = self.drop(self.embedding(text))
        embedded = embedded.permute(1,0,2)
        out = self.transformer_encoder(embedded)
        # out = out[0]
        out = torch.sum(out, dim=0)
        # print(out.shape)
        # exit(0)
        dense_outputs = self.fc(out)
        outputs = dense_outputs
        return outputs
