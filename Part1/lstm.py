from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.W_gx = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.W_gh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.randn(hidden_dim))
        self.W_ix = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.W_ih = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.randn(hidden_dim))
        self.W_fx = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.W_fh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.randn(hidden_dim))
        self.W_ox = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.W_oh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.randn(hidden_dim))
        self.W_ph = nn.Parameter(torch.randn(output_dim, hidden_dim))
        self.b_p = nn.Parameter(torch.randn(output_dim))
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Implementation here ...
        batch_size, seq_length, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for t in range(seq_length):
            x_t = x[:, t, :]
            g_t = torch.tanh(torch.mm(x_t, self.W_gx.t()) + torch.mm(h_t, self.W_gh.t()) + self.b_g)
            i_t = torch.sigmoid(torch.mm(x_t, self.W_ix.t()) + torch.mm(h_t, self.W_ih.t()) + self.b_i)
            f_t = torch.sigmoid(torch.mm(x_t, self.W_fx.t()) + torch.mm(h_t, self.W_fh.t()) + self.b_f)
            o_t = torch.sigmoid(torch.mm(x_t, self.W_ox.t()) + torch.mm(h_t, self.W_oh.t()) + self.b_o)
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

        p_t = torch.mm(h_t, self.W_ph.t()) + self.b_p
        return p_t
        





