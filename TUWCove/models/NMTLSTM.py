from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging

from TUWCove.utils import constants
###################
#Econder

class Encoder(nn.Module):
    def __init__(self, num_layers: int, bidirectional: bool, dropout: float, rnn_size: int, 
                 word_vec_dim: int, *args, **kwargs):
        """
        Args:
            num_layers (int): _description_
            bidirectional (bool): _description_
            dropout (float): _description_
            rnn_size (int): _description_
            word_vec_dim (int): _description_
        """
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_r = dropout
        self.dropout_l = nn.Dropout(p=dropout)
        self.dropout_l2 = nn.Dropout(p=dropout)
        # Calculate hidden size for bidirectional case
        self.hidden_size = rnn_size // 2 if bidirectional else rnn_size
        self.LSTM_input_size = word_vec_dim
        self.LSTM = nn.LSTM(
            input_size=self.LSTM_input_size, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_r,
            bidirectional=self.bidirectional,
            batch_first=True
        )

    def forward(self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]|None = None):
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
        emb = self.embedding(input)  # [batch_size x sequence_length x word_vec_dim]
        emb = self.dropout_l(emb)
        outputs, hidden_t = self.LSTM(emb, hidden)
        outputs = self.dropout_l2(outputs)

        return outputs, hidden_t


##################
# Decoder
    
class Decoder(nn.Module):
    
    def __init__(self, num_layers: int, bidirectional: bool, dropout: float, rnn_size:int,
                 word_vec_dim:int, dict_size:int, padding=constants.PAD, *args, **kwargs):
        """_summary_
        Args:
            num_layers (int): _description_
            bidirectional (bool): _description_
            dropout (float): _description_
            rnn_size (int): _description_
            word_vec_dim (int): _description_
            dict_size (int): _description_
            padding (_type_, optional): _description_. Defaults to constants.PAD.
        """
        super().__init__(*args, **kwargs)
        self.device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.input_size = word_vec_dim
        self.hidden_size = rnn_size
        self.embedding = nn.Embedding(num_embeddings = dict_size,
                                embedding_dim = self.input_size,
                                padding_idx = padding)
        self.dropout_r = dropout
        self.dropout_l1 = nn.Dropout(p=dropout)
        self.lstm_input_size = self.input_size + self.hidden_size
        self.LSTM = nn.LSTM(input_size = self.lstm_input_size,
                            hidden_size = self.hidden_size, 
                            num_layers = self.num_layers, 
                            dropout = self.dropout_r,
                            bidirectional = self.bidirectional,
                            batch_first=True)
        self.dropout_l2 = nn.Dropout(p=dropout)
        self.linear_attn_in = nn.Linear(in_features=rnn_size, out_features=rnn_size)
        self.linear_attn_out = nn.Linear(in_features=rnn_size*2, out_features=rnn_size)
        self.dropout_l3 = nn.Dropout(p=dropout)
        
    
    def forward(self, tgt_tensor: Tensor, encoder_out_vec_H: Tensor,
                encoder_hidden_out: Tuple[Tensor, Tensor]) -> Tensor:
        """_summary_
        Args:
            tgt_tensor (Tensor): _description_
            encoder_out_vec_H (Tensor): _description_
            encoder_hidden_out (Tuple[Tensor, Tensor]): _description_

        Returns:
            Tensor: _description_
        """
        # Initialize decoder states
        h_dec_tmin1 = encoder_hidden_out
        batch_size = tgt_tensor.shape[0]
        h_tilde_m1 = torch.zeros(batch_size, self.hidden_size, device=tgt_tensor.device)  # Ensure device compatibility
        tgt_word_embeddings = self.embedding(tgt_tensor)  # [batch_size x sentence_len x word_vec_dim]
        context_adj_states = []

        # Iterate over target embeddings timestep by timestep
        for emb_z_t in tgt_word_embeddings.split(1, dim=1):  # [batch_size x 1 x word_vec_dim]
            emb_z_t = self.dropout_l1(emb_z_t)

            # Concatenate input embedding with previous context state and pass through LSTM
            input_lstm = torch.cat([emb_z_t, h_tilde_m1.unsqueeze(1)], dim=-1)  # [batch_size x 1 x (word_vec_dim + rnn_size)]
            h_dec_t, hidden_cell = self.LSTM(input_lstm, h_dec_tmin1)  # h_dec_t: [batch_size x 1 x rnn_size]
            h_dec_t = self.dropout_l2(h_dec_t) 

            # Compute attention scores (equation 3)
            attention_l_in = self.linear_attn_in(h_dec_t)  # [batch_size x 1 x rnn_size]
            alpha_t_mul = torch.bmm(attention_l_in, encoder_out_vec_H.permute(0, 2, 1))  # [batch_size x 1 x sentence_len]
            alpha_t = F.softmax(alpha_t_mul, dim=-1)  # Normalize attention scores

            # Compute context vector (equation 4)
            alpha_T_H = torch.bmm(alpha_t, encoder_out_vec_H)  # [batch_size x 1 x rnn_size]

            # Combine decoder output and context vector
            context_combined = torch.cat([h_dec_t.squeeze(1), alpha_T_H.squeeze(1)], dim=1)  # [batch_size x (rnn_size*2)]
            context_adj_htilde = self.linear_attn_out(context_combined)  # [batch_size x rnn_size]

            # Apply dropout and activation to the adjusted context
            context_adj_htilde = self.dropout_l3(context_adj_htilde)
            context_adj_htilde = torch.tanh(context_adj_htilde)  # Final adjusted context

            # Update states for the next timestep
            h_tilde_m1 = context_adj_htilde
            h_dec_tmin1 = hidden_cell  # Update LSTM hidden states

            # Store the adjusted context state
            context_adj_states.append(context_adj_htilde)

        # Stack adjusted context states for all timesteps
        h_context_stack = torch.stack(context_adj_states, dim=1)  # [batch_size x sentence_len x rnn_size]
        return h_context_stack

    
class NMTModel(nn.Module):
    
    def __init__(self, encoder, decoder, rnn_size:int, tgt_dict_size:int, src_dict_size:int, word_vec_dim=300, use_glove: bool=False, glove_embeddings:Optional[Tensor]=None, padding=constants.PAD, dropout: float=0.1, *args, **kwargs):
        """
        Args:
            encoder (_type_): _description_
            decoder (_type_): _description_
            rnn_size (int): _description_
            tgt_dict_size (int): _description_
            src_dict_size (int): _description_
            word_vec_dim (int, optional): _description_. Defaults to 300.
            use_glove (bool, optional): _description_. Defaults to False.
            glove_embeddings (Optional[Tensor], optional): _description_. Defaults to None.
            padding (_type_, optional): _description_. Defaults to constants.PAD.
            dropout (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__(*args, **kwargs)
        if use_glove:
            assert glove_embeddings is not None
            self.embedding  = nn.Embedding.from_pretrained(glove_embeddings, freeze=True, padding_idx=padding)
        else:
            self.embedding = nn.Embedding(src_dict_size, embedding_dim=word_vec_dim, padding_idx=padding)
        self.output_vocab_size = tgt_dict_size
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(in_features=rnn_size, out_features=self.output_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    @classmethod
    def NMT_Glove(cls, src_dict_len:int, tgt_dict_len:int, glove_embeddings:Tensor, word_vec_dim:int=300, rnn_size:int=300, dropout:float=0.1, padding=constants.PAD):
        """Initialize the NMT model with precomputed embeddings."""

        encoder = Encoder(
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            rnn_size=rnn_size,
            word_vec_dim=word_vec_dim
        )
        decoder = Decoder(
            num_layers=2,
            bidirectional=False,
            dropout=dropout,
            rnn_size=rnn_size,
            word_vec_dim=word_vec_dim, 
            dict_size=tgt_dict_len,
            padding=padding
        )
        assert glove_embeddings.size(1) == word_vec_dim, f"Glove Vector length does not match dimension: {glove_embeddings.size(1)}, {word_vec_dim}"
        model = cls(
            encoder=encoder,
            decoder=decoder,
            rnn_size=rnn_size,
            src_dict_size= glove_embeddings.size(0),
            tgt_dict_size=tgt_dict_len,
            dropout=dropout,
            word_vec_dim=word_vec_dim,
            use_glove=True,
            glove_embeddings = glove_embeddings, #TODO load embedding_matrix
            padding=padding
        )
        return model
    
    @classmethod
    def NMT_rand_embs(cls, src_dict_len:int, tgt_dict_len:int, word_vec_dim:int=300, rnn_size:int=300, dropout:float=0.1, padding=constants.PAD):
        """Initialize the NMT model with precomputed embeddings."""

        encoder = Encoder(
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            rnn_size=rnn_size,
            word_vec_dim=word_vec_dim
        )
        decoder = Decoder(
            num_layers=2,
            bidirectional=False,
            dropout=dropout,
            rnn_size=rnn_size,
            word_vec_dim=word_vec_dim, 
            dict_size=tgt_dict_len,
            padding=padding
        )
        model = cls(
            encoder=encoder,
            decoder=decoder,
            rnn_size=rnn_size,
            src_dict_size= src_dict_len,
            tgt_dict_size=tgt_dict_len,
            dropout=dropout,
            word_vec_dim=word_vec_dim,
            use_glove=False,
            padding=padding
        )
        return model
        
    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
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
        tgt = input[1]
        
        embs = self.embedding(src)
        enc_out, enc_hidden = self.encoder(embs)
        
        enc_hidden = self._fix_enc_hidden(enc_hidden)
    
        decoder_out = self.decoder(tgt,enc_out, enc_hidden)
        decoder_out = self.dropout(decoder_out)
        logits = self.linear(decoder_out)
        dropped = self.dropout(logits)
        
        out = F.log_softmax(dropped, dim=-1)
        
        return out
