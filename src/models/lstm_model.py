
import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super(SelfAttention, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim = embedding_dim,
                                          num_heads = num_heads,
                                          batch_first = True)

    def forward(self, embeddings):
        output, _ = self.attn(embeddings, embeddings, embeddings)
        return output
        

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, embedding_dim, hidden_units, lstm_layers: int, attn_heads: int):
        super(LSTMClassifier, self).__init__()
 
        self.embeddings = nn.Embedding.from_pretrained(embeddings = embedding_matrix, freeze = True)

        self.attn_layer = SelfAttention(embedding_dim = embedding_dim,
                                        num_heads = attn_heads)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_units, 
                            num_layers = lstm_layers, 
                            batch_first = True, 
                            bidirectional = True)
        
        self.fc = nn.Sequential(nn.Linear(in_features = hidden_units * 2, out_features = hidden_units),
                                nn.Linear(in_features = hidden_units, out_features = 1),
                                nn.Dropout(0.2))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        embeddings = self.embeddings(x)

        attn_out = self.attn_layer(embeddings)
    
        lstm_out, _ = self.lstm(attn_out)
    
        lstm_out = lstm_out[:, -1, :]
        
        fc_out = self.fc(lstm_out).squeeze(1)

        output = self.sigmoid(fc_out)

        return output
