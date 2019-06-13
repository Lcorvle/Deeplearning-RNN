import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.gate = nn.Linear(input_size + hidden_size, cell_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, cell):
        output = input
        for i in range(self.num_layers):
            combined = torch.cat((output, hidden), 1)
            f_gate = self.gate(combined)
            f_gate = self.sigmoid(f_gate)
            i_gate = self.gate(combined)
            i_gate = self.sigmoid(i_gate)
            o_gate = self.gate(combined)
            o_gate = self.sigmoid(o_gate)
            cell_helper = self.gate(combined)
            cell_helper = self.tanh(cell_helper)
            cell = torch.add(torch.mul(cell, f_gate), torch.mul(cell_helper, i_gate))
            hidden = torch.mul(self.tanh(cell), o_gate)
            output = self.output(hidden)
            output = self.softmax(output)
        return output, hidden, cell

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def initCell(self):
        return Variable(torch.zeros(1, self.cell_size))