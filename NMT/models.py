import torch
import torch.nn as nn
from torch.autograd import Variable

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
            #TODO check v
            #This embedding layer should have _freeze = True 
            # when passed or when declared? or when optimizers is defined?
        self.LSTM = nn.LSTM(input_size= self.word_vec_dim, 
                            hidden_size= self.hidden_size, 
                            num_layers= self.num_layers, 
                            dropout= self.dropout_r, 
                            bidirectional= self.bidirectional)
        #TODO create and insert new dropout layer
        
    def forward(self, input, hidden=None):
        if self.embedding is None:
            emb = input
        else:
            emb = self.embedding(input)
        emb = self.dropout_l(emb)
        outputs, hidden_t = self.LSTM(emb, hidden)
        #TODO create and inser new dropout layer
        return hidden_t, outputs
    
    
class Decoder(nn.Module):
    #  Dropout with ratio 0.2 was applied to the inputs and outputs of 
    # all layers of the encoder and decoder.
    
    def __init__(self, num_layers: int, bidirectional: bool, dropout: float, rnn_size:int, word_vec_dim:int, dict_size:int, padding=constants.PAD, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_r = dropout
        self.dropout_l = nn.Dropout(p=dropout)
        self.input_size = word_vec_dim
        self.hidden_size = rnn_size
        self.embedding = nn.Embedding(num_embeddings = dict_size, # TODO what other size?
                                embedding_dim = self.input_size,
                                padding_idx = padding)
        
        self.LSTM = nn.LSTM(input_size = self.input_size,
                            hidden_size = self.hidden_size, 
                            num_layers = self.num_layers, 
                            dropout = self.dropout_r,
                            bidirectional = self.bidirectional)
        self.attn = GlobalAttention(rnn_size)
        
        ##TODO should this be separate? vvvvvv
        self.linear = nn.Linear(in_features=rnn_size, out_features=dict_size)
        self.softmax = nn.Softmax()#?
        
        
    def forward(self, input, hidden, context, init_output):
        emb = self.embedding(input)
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = self.dropout_l(emb_t)
            output = self.dropout_l(output)
            #TODO 1
            # ValueError: LSTM: Expected input to be 2D or 3D, got 1D instead
            # FIX remove squeeze 
            #emb_t = emb_t.squeeze(0)
            output, hidden = self.LSTM(emb_t, hidden)
            #TODO create and inser new dropout layer
            output, attn = self.attn(output, context.t()) #(4)
            outputs += [output]
        
        outputs = torch.stack(outputs)
        return outputs, hidden, attn
    
class NMTModel(nn.Module):
    
    def __init__(self, encoder, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.bidirectional:
            #TODO 3
            # fix_hidden doesn't have the right dimensions
            # IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)).transpose(1, 2).contiguous().view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h
        
    def forward(self, input):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)
        
        #TODO 2 without squeeze
        #RuntimeError: Expected hidden[0] size (1, 1, 50), got [2, 1, 50]
        #Fix return fix_hidden
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        
        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden, context, init_output)

        
        return out