import math

import torch
import torch.nn as nn
from dpp.utils.gumbel import gumbel_softmax, my_softmax
from dpp.utils.helper import get_cuda, get_offdiag_indices
from torch.autograd import Variable
from torch.nn import functional as F
from tyche.utils.helper import quadratures, create_instance

_EPS = 1e-10


class RNN(nn.Module):
    def __init__(self, input_size, **kwargs):
        super(RNN, self).__init__()
        kwargs['cell_type']['args']['batch_first'] = True
        kwargs['cell_type']['args']['input_size'] = input_size

        self.rnn = create_instance('cell_type', kwargs)
        self.hidden_state = None

    def forward(self, input):
        t, seq_len = input  # B, T, D

        _ix = seq_len.nonzero().view(-1)
        x = torch.nn.utils.rnn.pack_padded_sequence(t[_ix], seq_len[_ix], batch_first=True)
        hidden_state = self._get_hidden_states(_ix)
        output, hidden_state = self.rnn(x, hidden_state)
        self._update_hidden_state(hidden_state, _ix)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=t.size(1))

        return output

    def _get_hidden_states(self, _ix):
        if type(self.rnn) is nn.LSTM:
            return tuple(x[:, _ix] for x in self.hidden_state)
        else:
            return self.hidden_state[:, _ix]

    def _update_hidden_state(self, hidden_state, _ix):
        if type(self.rnn) is nn.LSTM:
            self.hidden_state[0][:, _ix] = hidden_state[0]
            self.hidden_state[1][:, _ix] = hidden_state[1]
        else:
            self.hidden_state[:, _ix] = hidden_state

    def initialize_hidden_state(self, batch_size: int, device: any):
        if type(self.rnn) is nn.LSTM:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            self.hidden_state = (h.to(device), c.to(device))
        else:
            self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def reset_history(self):
        if type(self.rnn) is nn.LSTM:
            self.hidden_state = tuple(x.detach() for x in self.hidden_state)
        else:
            self.hidden_state.detach_()

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

    @property
    def num_layers(self):
        return self.rnn.num_layers