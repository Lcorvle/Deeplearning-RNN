import torch
import torch.nn as nn
from torch.nn import Parameter
from enum import IntEnum
import torch.nn.functional as F
import pdb


class Dim(IntEnum):
    batch = 1
    seq = 0
    feature = 2


class NaiveLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # input gate
        self.W_ii0 = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ii = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size))
        # forget gate
        self.W_if0 = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_if = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))
        # helper gate
        self.W_ig0 = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ig = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(torch.Tensor(hidden_size))
        # output gate
        self.W_io0 = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_io = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        """Assumes x is of shape (sequence, batch, feature)"""
        seq_sz, bs, _ = x.size()

        for l in range(self.num_layers):
            hidden_seq = []
            if init_states is None:
                h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
            else:
                h_t, c_t = init_states
            for t in range(seq_sz):  # iterate over the time steps
                x_t = x[t, :, :]
                if l == 0:
                    i_t = torch.sigmoid(x_t @ self.W_ii0 + h_t @ self.W_hi + self.b_i)
                    f_t = torch.sigmoid(x_t @ self.W_if0 + h_t @ self.W_hf + self.b_f)
                    g_t = torch.tanh(x_t @ self.W_ig0 + h_t @ self.W_hg + self.b_g)
                    o_t = torch.sigmoid(x_t @ self.W_io0 + h_t @ self.W_ho + self.b_o)
                else:
                    i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
                    f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
                    g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
                    o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
                hidden_seq.append(h_t.unsqueeze(Dim.batch))
            # pdb.set_trace()
            hidden_seq = torch.cat(hidden_seq, dim=Dim.batch).transpose(Dim.batch, Dim.seq).contiguous()
            x = hidden_seq
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        # hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


class OptimizedLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # gates
        self.W_ii0 = Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.W_ii = Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = Parameter(torch.Tensor(hidden_size * 4))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        """Assumes x is of shape (sequence, batch, feature)"""
        seq_sz, bs, _ = x.size()

        for l in range(self.num_layers):
            hidden_seq = []
            if init_states is None:
                h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
            else:
                h_t, c_t = init_states
            for t in range(seq_sz):  # iterate over the time steps
                x_t = x[t, :, :]
                if l == 0:
                    gates = x_t @ self.W_ii0 + h_t @ self.W_hi + self.bias
                else:
                    gates = x_t @ self.W_ii + h_t @ self.W_hi + self.bias
                i_t, f_t, g_t, o_t = (
                    torch.sigmoid(gates[:, :self.hidden_size]),  # input
                    torch.sigmoid(gates[:, self.hidden_size:self.hidden_size * 2]),  # forget
                    torch.tanh(gates[:, self.hidden_size * 2:self.hidden_size * 3]),
                    torch.sigmoid(gates[:, self.hidden_size * 3:]),  # output
                )
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
                hidden_seq.append(h_t.unsqueeze(Dim.batch))
            # pdb.set_trace()
            hidden_seq = torch.cat(hidden_seq, dim=Dim.batch).transpose(Dim.batch, Dim.seq).contiguous()
            x = hidden_seq
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        # hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


class MyAttention(nn.Module):
    def __init__(self, hidden_size):
        super(MyAttention, self).__init__()
        self.attn_fn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            nn.init.zeros_(p.data)

    def forward(self, x): #x: [S B H]
        z = torch.tanh(self.attn_fn(x))
        z = z.transpose(0, 1) #[B S H]
        v = self.v.repeat(x.size(1), 1).unsqueeze(2)  # [B H 1]
        z = torch.bmm(z, v).squeeze(2)  # [B S]
        attn_scores = F.log_softmax(z, dim=1)
        context = torch.matmul(attn_scores.unsqueeze(dim=1),
                               x.transpose(0, 1)).squeeze()
        return context


class MyDecoder(nn.Module):
    def __init__(self, hidden_size, nvoc, isAttention=True):
        super(MyDecoder, self).__init__()
        self.attention = MyAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, nvoc)
        self.isAttn = isAttention

    def forward(self, x):
        context = torch.cat(tuple([self.attention(x[0 : i, :, :]).unsqueeze(0)
                                       for i in range(1, x.size(0) + 1)]), dim=0) if self.isAttn else x
        z = self.fc(context)
        z = F.log_softmax(z, dim=1)
        return z