""" Main model implementation script """

import torch
import torch.nn as nn

class LSTMTimeSeries(nn.Module):
    """Simple LSTM implementation for time series analysis"""

    def __init__(self, input_size : int, hidden_size : int, num_layers : int,
                 output_size : int, dropout : float = 0.2):
        super(LSTMTimeSeries, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout = dropout)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0)) # tensor (batch, seq.length, hidden_size)

        out = self.fc(out[:, -1, :])
        return out