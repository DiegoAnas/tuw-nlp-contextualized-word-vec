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
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)  #TODO check if is the right dimension
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        #TODO from dim to dim // 2
        self.tanh = nn.Tanh()
        self.mask = None
        self.dot = dot

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: h_dec_t = lstm_output [1 x sentence_len x dim]
        context: encoder output vector [batch x sentence_len x dim] 
        """
        #eqn 3
        target_t = self.linear_in(input) # batch(1) x sentenceL x rnn_size 
        # Get attention
        
        #attention = torch.matmul(encoder_out_tuple[1].squeeze(0),target_T.squeeze(0).t())
        
        attention = torch.matmul(context[1].squeeze(0),target_t.squeeze(0).t())
        #attn = torch.bmm(context, target_t).squeeze(2)  # sen_len x sen_len
        attn = self.sm(attention)  # alpha_t
        
        # Eqn 4
        # weightedContext # [sen_len x sen_len] x [sen_len x dim] [sen_len x dim]
        # or #H transpose * alpha t
        alpha_t_H = torch.matmul(attn, target_t.squeeze(0)) 
          
        context_combined = torch.cat((alpha_t_H, input.squeeze(0)), dim=1)      

        contextOutput = self.tanh(self.linear_out(context_combined))

        return contextOutput, attn
