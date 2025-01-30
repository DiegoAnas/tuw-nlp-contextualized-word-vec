import torch
import torch.nn as nn
from BERTembedding import BERTembedding

class BCNWithBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", bilstm_hidden_size=256, fc_hidden_size=128, num_labels=2):
        super().__init__()
        
        # Load BERT embeddings model from nlptechbook/BERTembeddings
        self.bert_embedder = BERTembedding(model_name)
        
        # BiLSTM Encoder
        self.bilstm = nn.LSTM(input_size=768, hidden_size=bilstm_hidden_size, batch_first=True, bidirectional=True)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(bilstm_hidden_size * 2, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, num_labels)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, input_texts):
        """
        Forward pass through the BCN model with embeddings from BERTembedding.
        """
        # Get embeddings from BERTembedding
        bert_embeddings = self.bert_embedder(input_texts)
        
        # Pass through BiLSTM
        lstm_out, _ = self.bilstm(bert_embeddings.unsqueeze(1))
        lstm_out = lstm_out[:, -1, :]  # Take the last output of BiLSTM
        
        # Fully Connected Layers
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

