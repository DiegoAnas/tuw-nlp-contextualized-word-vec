"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                          a

Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

"""

import torch
import torch.nn as nn
import math

class GlobalAttention(nn.Module):
    def __init__(self, dim, dot=False):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False) # paper has bias!
        self.sm = nn.Softmax(dim=1)  #TODO check if is the right dimension
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        #TODO from dim to dim // 2
        self.tanh = nn.Tanh()
        self.mask = None
        self.dot = dot

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, h_dec_t, enc_h_out):
        """
        h_dec_t: h_dec_t = lstm_output [batch x sentence len x dim(rnn_size)]
        enc_h_out: encoder output vector [batch x sentence_len x dim(rnn_size)]
        """
        assert h_dec_t.shape == enc_h_out.shape, f"Expected equal shapes, got {h_dec_t.shape};{enc_h_out.shape}"
        #eqn 3
        target_t = self.linear_in(h_dec_t) # batch(1) x sentenceL x rnn_size 
        # Get attention
                
        attention = torch.matmul(enc_h_out[1].squeeze(0),target_t.squeeze(0).t())
        #attn = torch.bmm(context, target_t).squeeze(2)  # sen_len x sen_len
        attn = self.sm(attention)  # alpha_t
        
        # Eqn 4
        # weightedContext # [sen_len x sen_len] x [sen_len x dim] [sen_len x dim]
        # or #H transpose * alpha t
        alpha_t_H = torch.matmul(attn, target_t.squeeze(0)) 
          
        context_combined = torch.cat((alpha_t_H, h_dec_t.squeeze(0)), dim=1)      

        contextOutput = self.tanh(self.linear_out(context_combined))

        return contextOutput, attn
