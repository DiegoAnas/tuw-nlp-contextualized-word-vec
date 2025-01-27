from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging

from NMT import constants
from cove import CoVeEncoder
###################
#Econder

class Encoder(nn.Module):
    def __init__(self, num_layers: int, bidirectional: bool, dropout: float, rnn_size: int, 
                 word_vec_dim: int, dict_size: int, vocab: dict, glove_embeddings: dict, 
                 padding=constants.PAD, freeze_embeddings=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_r = dropout
        self.dropout_l = nn.Dropout(p=dropout)
        self.dropout_l2 = nn.Dropout(p=dropout)

        # Calculate hidden size for bidirectional case
        self.hidden_size = rnn_size // 2 if bidirectional else rnn_size
        self.word_vec_dim = word_vec_dim

        # Initialize the embedding layer with GloVe embeddings
        embedding_matrix = self._create_embedding_matrix(vocab, glove_embeddings, word_vec_dim)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embeddings, padding_idx=padding)

        # Initialize CoVe encoder
        self.cove = CoVeEncoder(input_size=self.word_vec_dim, hidden_size=rnn_size)

        # Initialize the main LSTM
        self.LSTM = nn.LSTM(
            input_size=rnn_size,  # CoVe outputs should match the LSTM input
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_r,
            bidirectional=self.bidirectional,
            batch_first=True
        )

    def _create_embedding_matrix(self, vocab: dict, glove_embeddings: dict, embedding_dim: int) -> torch.Tensor:
        """
        Create an embedding matrix for the vocabulary using GloVe embeddings.
        Args:
            vocab (dict): Vocabulary mapping (word -> index).
            glove_embeddings (dict): Preloaded GloVe embeddings.
            embedding_dim (int): Dimension of GloVe embeddings (e.g., 300).
        Returns:
            torch.Tensor: An embedding matrix of shape (vocab_size, embedding_dim).
        """
        vocab_size = len(vocab)
        embedding_matrix = torch.zeros((vocab_size, embedding_dim))
        for word, idx in vocab.items():
            if word in glove_embeddings:
                embedding_matrix[idx] = glove_embeddings[word]
            else:
                embedding_matrix[idx] = torch.rand(embedding_dim)  # Random for unknown words
        return embedding_matrix

    def forward(self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        """
        Forward pass for the Encoder.
        Args:
            input (torch.Tensor): Input tensor of token indices [batch_size x sequence_length].
            hidden (Tuple[torch.Tensor, torch.Tensor], optional): Initial hidden states for LSTM.
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - outputs: LSTM outputs [batch_size x sequence_length x hidden_size].
                - hidden_t: Final hidden states from LSTM.
        """
        # Pass through the embedding layer
        emb = self.embedding(input)  # [batch_size x sequence_length x word_vec_dim]
        emb = self.dropout_l(emb)

        # Pass through the CoVe encoder
        cove_outputs = self.cove(emb)  # [batch_size x sequence_length x rnn_size]

        # Pass through the main LSTM
        outputs, hidden_t = self.LSTM(cove_outputs, hidden)
        outputs = self.dropout_l2(outputs)

        return outputs, hidden_t


##################
# Decoder
    
class Decoder(nn.Module):
    #  Dropout with ratio 0.2 was applied to the inputs and outputs of 
    # all layers of the encoder and decoder.
    
    def __init__(self, num_layers: int, bidirectional: bool, dropout: float, rnn_size:int, word_vec_dim:int, dict_size:int, padding=constants.PAD, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO ? self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.input_size = word_vec_dim
        self.hidden_size = rnn_size
        self.embedding = nn.Embedding(num_embeddings = dict_size, # TODO what other size?
                                embedding_dim = self.input_size,
                                padding_idx = padding)
        self.dropout_r = dropout
        self.dropout_l1 = nn.Dropout(p=dropout)
        #self.decoder_lstm_input = embedding_size + hidden_size self.input_size + self.hidden_size
        self.LSTM = nn.LSTM(input_size = self.input_size,
                            hidden_size = self.hidden_size, 
                            num_layers = self.num_layers, 
                            dropout = self.dropout_r,
                            bidirectional = self.bidirectional,
                            batch_first=True)
        self.dropout_l2 = nn.Dropout(p=dropout)

        self.linear_attn_in = nn.Linear(in_features=rnn_size, out_features=rnn_size)
        self.linear_attn_out = nn.Linear(in_features=rnn_size*2, out_features=rnn_size)
        
        
    
def forward(self, tgt_tensor: Tensor, encoder_out_vec_H: Tensor, encoder_hidden_out: Tuple[Tensor, Tensor]) -> Tensor:

    # Initialize decoder states
    h_dec_tmin1 = encoder_hidden_out
    batch_size = tgt_tensor.shape[0]
    h_tilde_m1 = torch.zeros(batch_size, self.hidden_size, device=tgt_tensor.device)  # Ensure device compatibility
    tgt_word_embeddings = self.embedding(tgt_tensor)  # [batch_size x sentence_len x word_vec_dim]

    # Store adjusted context states for each timestep
    context_adj_states = []

    # Iterate over target embeddings timestep by timestep
    for emb_z_t in tgt_word_embeddings.split(1, dim=1):  # [batch_size x 1 x word_vec_dim]
        # Apply dropout to input embeddings
        emb_z_t = self.dropout_l1(emb_z_t)

        # Concatenate input embedding with previous context state
        input_lstm = torch.cat([emb_z_t, h_tilde_m1.unsqueeze(1)], dim=-1)  # [batch_size x 1 x (word_vec_dim + rnn_size)]

        # Pass through LSTM
        h_dec_t, hidden_cell = self.LSTM(input_lstm, h_dec_tmin1)  # h_dec_t: [batch_size x 1 x rnn_size]

        # Apply dropout to LSTM outputs
        h_dec_t = self.dropout_l2(h_dec_t)  # h_dec_t: [batch_size x 1 x rnn_size]

        # Compute attention scores (equation 3 in paper)
        attention_l_in = self.linear_attn_in(h_dec_t)  # [batch_size x 1 x rnn_size]
        alpha_t_mul = torch.bmm(attention_l_in, encoder_out_vec_H.permute(0, 2, 1))  # [batch_size x 1 x sentence_len]
        alpha_t = F.softmax(alpha_t_mul, dim=-1)  # Normalize attention scores

        # Compute context vector (equation 4 in paper)
        alpha_T_H = torch.bmm(alpha_t, encoder_out_vec_H)  # [batch_size x 1 x rnn_size]

        # Combine decoder output and context vector
        context_combined = torch.cat([h_dec_t.squeeze(1), alpha_T_H.squeeze(1)], dim=1)  # [batch_size x (rnn_size*2)]
        context_adj_htilde = self.linear_attn_out(context_combined)  # [batch_size x rnn_size]

        # Apply dropout and activation to the adjusted context
        context_adj_htilde = self.dropout_l4(context_adj_htilde)
        context_adj_htilde = torch.tanh(context_adj_htilde)  # Final adjusted context

        # Update states for the next timestep
        h_tilde_m1 = context_adj_htilde
        h_dec_tmin1 = hidden_cell  # Update LSTM hidden states

        # Store the adjusted context state
        context_adj_states.append(context_adj_htilde)

    # Stack adjusted context states for all timesteps
    h_context_stack = torch.stack(context_adj_states, dim=1)  # [batch_size x sentence_len x rnn_size]
    return h_context_stack
        
class CoVeEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        A CoVe encoder using a pre-trained Bi-LSTM.

        Args:
            input_size (int): Dimension of the input embeddings (e.g., 300 for GloVe).
            hidden_size (int): Hidden size of the Bi-LSTM.
        """
        super(CoVeEncoder, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, inputs):
        """
        Forward pass for CoVe.

        Args:
            inputs (Tensor): [batch_size, seq_len, embedding_dim].

        Returns:
            Tensor: Contextualized embeddings from the Bi-LSTM.
        """
        outputs, _ = self.bilstm(inputs)
        return outputs

    
class NMTModel(nn.Module):
    
    def __init__(self, encoder, decoder, rnn_size:int, tgt_dict_size:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_vocab_size = tgt_dict_size
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(in_features=rnn_size, out_features=self.output_vocab_size)
        #TODO add dropout module
    self.dropout = nn.Dropout(p=kwargs.get('dropout', 0.1))
        
    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        #TODO when not batched, this gives error
        if len(h[0].shape) == 2:  # When not batched
            h = (h[0].unsqueeze(1), h[1].unsqueeze(1))
        if self.encoder.bidirectional:
            assert len(h) == 2 , f"enc_hidden tuple length 2 expected, got {len(h)}.\n{self.encoder}"
            assert len(h[0].shape) == 3, f"expected [(layers*directions), batch, dim] enc_hidden, got {h[0].shape}.\n{self.encoder}"
            return (h[0].view(h[0].shape[0]//2, h[0].shape[1], h[0].shape[2]*2),
                    h[1].view(h[1].shape[0]//2, h[1].shape[1], h[1].shape[2]*2))
        else:
            return h
        
    def forward(self, input:Tuple):
        src = input[0]
        tgt = input[1]  # ¿exclude last target from inputs?
        enc_out, enc_hidden = self.encoder(src)
        
        enc_hidden = self._fix_enc_hidden(enc_hidden)
    
        decoder_out = self.decoder(tgt,enc_out, enc_hidden)
        decoder_out = self.dropout(decoder_out)
        out = self.linear(decoder_out)
        #TODO add dropout op?
        
        out = F.log_softmax(out, dim=-1)
        
        return out