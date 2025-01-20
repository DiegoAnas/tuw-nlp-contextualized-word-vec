import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from NMT import constants
from NMT.modules import GlobalAttention 


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
        #TODO test LSTM batched to get a 32x hidden states
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
        self.LSTM = nn.LSTM(input_size = self.input_size,
                            hidden_size = self.hidden_size, 
                            num_layers = self.num_layers, 
                            dropout = self.dropout_r,
                            bidirectional = self.bidirectional,
                            batch_first=True)
        self.dropout_l2 = nn.Dropout(p=dropout)
        #TODO test LSTM batched to get a 32x hidden states to use on attention module
        # and that way can use torch.bmm matrix multiplication (instead of matmul)
        self.attn = GlobalAttention(rnn_size)
        self.linear = nn.Linear(in_features=rnn_size, out_features=dict_size)
        
        
    def forward(self, input, hidden, context):
        """_summary_

        Args:
            input (_type_): target tensor [batch x sentence_len x dim]
            hidden (_type_): final hidden state of the encoder [layers x sentence_len x (directions*dim)][2]
            context (_type_): output of the encoder 

        Returns:
            _type_: _description_
        """
        emb = self.embedding(input)
        context_adj_states = []
        for emb_t in emb.split(1): #iterate over batch
            emb_t = self.dropout_l1(emb_t) # 1 x sentence_length x dimension
            #if self.input_feed:
            #    emb_t = torch.cat([emb_t, output], 1)
            #h_dec_t, cell_t
            h_dec_t, hidden = self.LSTM(emb_t, hidden)
            h_dec_t = self.dropout_l2(h_dec_t)
            #hidden num_layers*proj_size or *hidden_size
            context_adj_ht, attn = self.attn(h_dec_t, context.t()) #(4)
            context_adj_states += [context_adj_ht]
        
        #TODO test
        h_context_stack = torch.stack(context_adj_states)
        outputs = F.log_softmax(self.linear(h_context_stack))
        return outputs, hidden, attn
    
class NMTModel(nn.Module):
    
    def __init__(self, encoder, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
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
            assert len(h) == 2 , f"enc_hidden tuple dimension unexpected, got {len(h)}"
            assert len(h[0].shape) == 3, f"enc_hidden dimension unexpected, got {h[0].shape}"
            return (h[0].view(h[0].shape[0]//2, h[0].shape[1], h[0].shape[2]*2),
                    h[1].view(h[1].shape[0]//2, h[1].shape[1], h[1].shape[2]*2))
        else:
            return h
        
    def forward(self, input):
        src = input[0]
        tgt = input[1]  # ¿exclude last target from inputs?
        context, enc_hidden = self.encoder(src)
        
        # if input_feed
        #init_output = self.make_init_decoder_output(context)
        
        enc_hidden = self._fix_enc_hidden(enc_hidden)
        
        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden, context)

        
        return out