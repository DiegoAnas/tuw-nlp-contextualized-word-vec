from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging

from NMT import constants

class Encoder(nn.Module):
    #  Dropout with ratio 0.2 was applied to the inputs and outputs of 
    # all layers of the encoder and decoder.
    
    def __init__(self, num_layers: int, bidirectional: bool, dropout: float, rnn_size:int, word_vec_dim:int, dict_size:int,emb_layer=None, padding=constants.PAD, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_r = dropout
        self.dropout_l = nn.Dropout(p=dropout)
        self.dropout_l2 = nn.Dropout(p=dropout)
        assert (bidirectional and (rnn_size % 2 == 0)) or (not bidirectional)
        if bidirectional:
            self.hidden_size = rnn_size // 2
        else:
            self.hidden_size = rnn_size
        self.word_vec_dim = word_vec_dim
        if emb_layer is None:
            self.embedding = nn.Embedding(num_embeddings = dict_size,
                                        embedding_dim = self.word_vec_dim,
                                        padding_idx = padding)
        else:
            self.embedding = emb_layer
            #TODO if emb_layer is passed make sure passed data uses GLOVE indexes
            #This embedding layer should have _freeze = True when declared
        self.LSTM = nn.LSTM(input_size= self.word_vec_dim, 
                            hidden_size= self.hidden_size, 
                            num_layers= self.num_layers, 
                            dropout= self.dropout_r, 
                            bidirectional= self.bidirectional,
                            batch_first=True)
        self.dropout_l2 = nn.Dropout(p=dropout)
        
    def forward(self, input, hidden=None):
        if self.embedding is None:
            emb = input
        else:
            emb = self.embedding(input)
        emb = self.dropout_l(emb)
        
        outputs, hidden_t = self.LSTM(emb, hidden)
        #TODO check if output with batch is the right size
        outputs = self.dropout_l2(outputs)
        return outputs, hidden_t
    
##################
# Decoder
    
class Decoder(nn.Module):
    #  Dropout with ratio 0.2 was applied to the inputs and outputs of 
    # all layers of the encoder and decoder.
    
    def __init__(self, num_layers: int, bidirectional: bool, dropout: float, rnn_size:int, word_vec_dim:int, dict_size:int, padding=constants.PAD, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        
        
    
    def forward(self, tgt_tensor:Tensor, encoder_out_vec_H:Tensor, encoder_hidden_out: Tuple[Tensor, Tensor], )-> Tensor:
        """LSTM and global attention mechanism

        Args:
            input (_type_): target tensor [batch x sentence_len]
            encoder_out_vec_H (_type_): output of the encoder [batch x sentence len x rnn_size]
            encoder_hidden_out (_type_): final hidden state of the encoder 2x[layers x batch x rnn_size]
        Returns:
            Tensor: [batch, sen_len, rnn_size]
        """
        h_dec_tmin1 = encoder_hidden_out
        batch_size = tgt_tensor.shape[0]
        sen_len = tgt_tensor.shape[1]
        h_tilde_m1 = torch.zeros(encoder_out_vec_H.shape[0], self.hidden_size)
        tgt_word_embeddings = self.embedding(tgt_tensor) # [batch, sentence_len, word_vec_dim]
        context_adj_states = []
        for emb_z_t in tgt_word_embeddings.split(1,dim=1): # iterate over words
            emb_z_t = self.dropout_l1(emb_z_t) # emb_z_t [batch, 1, word_vec_dim]
            h_dec_t, hidden_cell = self.LSTM(emb_z_t, h_dec_tmin1) 
            h_dec_t = self.dropout_l2(h_dec_t) # h_dec_t.shape == [1, sen_len, rnn_size]
            # hidden == [2][layers x batch x rnn_size]
            
            # equation 3
            attention_l_in = self.linear_attn_in(h_dec_t)
            # [batch, 1, rnn] x [batch, sen_len, rnn]
            alpha_t_mul = torch.bmm(attention_l_in, encoder_out_vec_H.permute(0,2,1))
            alpha_t = F.softmax(alpha_t_mul, dim=-1)
            # alpha_t [sen_len, 1, batch]

            # equation 4
            # Transpose H multiply by attention alpha_t
            # alpha_t[batch, 1, sen_len] X H[batch, sen_len, rnn_size]  = batch x 1 x rnn
            alpha_T_H = torch.bmm(alpha_t, encoder_out_vec_H)
            context_combined = torch.cat([h_dec_t.squeeze(1), alpha_T_H.squeeze(1)], 1)
            context_adj_htilde = self.linear_attn_out(context_combined)
            #TODO context_adj_htilde = self.dropout_l4(context_adj_htilde)
            context_adj_htilde = torch.tanh(context_adj_htilde)

            h_tilde_m1 = context_adj_htilde
            h_dec_tmin1 = hidden_cell # paper says it should the output of the LSTM not the hidden states
            print(context_adj_htilde.shape)
            context_adj_states.append(context_adj_htilde) #h_tildes
        
        h_context_stack = torch.stack(context_adj_states,dim=1)
        return h_context_stack
    
class NMTModel(nn.Module):
    
    def __init__(self, encoder, decoder, rnn_size:int, tgt_dict_size:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(in_features=rnn_size, out_features=tgt_dict_size)
        #TODO add dropout module?
        
    def make_init_decoder_output(self, context) -> torch.Tensor:
        """Initialize an all zeros initial tensor
        Args:
            context (torch.Tensor): with shape [batch_size, sentence length,word_vec_dim]
        Returns:
            torch.Tensor: _description_
        """
        batch_size = context.size(1) #TODO Wrong this is sentence length
        return torch.zeros(batch_size, self.decoder.hidden_size)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.bidirectional:
            assert len(h) == 2 , f"enc_hidden tuple length 2 expected, got {len(h)}.\n{self.encoder}"
            assert len(h[0].shape) == 3, f"expected 3D enc_hidden, got {h[0].shape}.\n{self.encoder}"
            return (h[0].view(h[0].shape[0]//2, h[0].shape[1], h[0].shape[2]*2),
                    h[1].view(h[1].shape[0]//2, h[1].shape[1], h[1].shape[2]*2))
        else:
            return h
        
    def forward(self, input):
        src = input[0]
        tgt = input[1]  # ¿exclude last target from inputs?
        enc_out, enc_hidden = self.encoder(src)
        
        enc_hidden = self._fix_enc_hidden(enc_hidden)
    
        decoder_out = self.decoder(tgt,enc_out, enc_hidden)

        out = self.linear(decoder_out)
        #TODO add dropout op?
        out = F.softmax(out, dim=-1)
        
        return out