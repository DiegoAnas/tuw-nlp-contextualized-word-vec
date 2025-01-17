import torch
import torch.nn as nn
from torch.autograd import Variable

from NMT import constants
from NMT.modules import GlobalAttention 

class Encoder(nn.Module):
    
    def __init__(self, num_layers: int, bidirectional: bool, dropout: float, rnn_size:int, dict_size:int,emb_layer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p=dropout)
        self.rnn_size = rnn_size
        self.hidden_size = rnn_size
        if emb_layer is None:
            self.embedding = nn.Embedding(num_embeddings=dict_size,
                                    embedding_dim=rnn_size,
                                    padding_idx=constants.PAD)
        else:
            self.embedding = emb_layer
        self.LSTM = nn.LSTM(input_size=rnn_size, hidden_size=rnn_size, num_layers=num_layers, 
                            dropout=dropout, bidirectional=bidirectional)
        
    def forward(self, input, hidden=None):
        if self.embedding is None:
            emb = input
        else:
            emb = self.embedding(input)
        emb = self.dropout(emb)
        outputs, hidden_t = self.LSTM(emb, hidden)
        return hidden_t, outputs
    
    
class Decoder(nn.Module):
    
    def __init__(self, num_layers: int, bidirectional: bool, dropout: float, rnn_size:int , dict_size:int, padding=constants.PAD, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        self.rnn_size = rnn_size
        self.hidden_size = rnn_size
        self.embedding = nn.Embedding(num_embeddings=dict_size, # TODO what other size?
                                embedding_dim=rnn_size,
                                padding_idx=padding) #TODO adapt padding constant
        
        self.LSTM = nn.LSTM(input_size=rnn_size, hidden_size=rnn_size, 
                            num_layers=num_layers, dropout=dropout,bidirectional=bidirectional)
        self.attn = GlobalAttention(rnn_size)
        
        ##TODO should this be separate? vvvvvv
        self.linear = nn.Linear(in_features=rnn_size, out_features=dict_size)
        self.softmax = nn.Softmax()#?
        
        
    def forward(self, input, hidden, context, init_output):
        emb = self.embedding(input)
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = self.dropout(emb_t)
            output = self.dropout(output)
            #emb_t = emb_t.squeeze(0)
            #TODO FIX
            # ValueError: LSTM: Expected input to be 2D or 3D, got 1D instead
            output, hidden = self.LSTM(emb_t, hidden)
            output, attn = self.attn(output, context.t()) #(4)
            output = self.dropout(output)
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
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h
        
    def forward(self, input):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)

        #enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
        #              self._fix_enc_hidden(enc_hidden[1]))

        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden, context, init_output)

        
        return out