import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
from RNN.LSTM import LSTM
from utils.Param_transfer import get_sparsity


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, time_major=False):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.time_major = time_major
        #self.h_sparsity = h_sparsity
        self.lstm1 = LSTM(input_size, hidden_size, time_major)
        self.lstm2 = LSTM(hidden_size, hidden_size, time_major)
        #self.lstm3 = LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.init_weights()

    def forward(self, x, states, h_sparsity=0.):
        
        final_state = []
        layer1, state = self.lstm1(x, states[0][0], states[1][0], h_sparsity=h_sparsity)
        final_state.append(state)
        layer2, state = self.lstm2(layer1, states[0][1], states[1][1], h_sparsity=h_sparsity)
        final_state.append(state)
        if not self.time_major:
            out = self.fc(layer2[:, -1, :])
        else:
            out = self.fc(layer2[-1, :, :])
        return out, tuple(final_state)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
       
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                weight.new_zeros(self.num_layers, bsz, self.hidden_size))
       
    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)