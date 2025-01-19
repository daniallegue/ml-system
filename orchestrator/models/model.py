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


class GRUTimeSeries(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUTimeSeries, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)  # out: (batch, seq_length, hidden_size)
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
        return out


class FeedforwardTimeSeries(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(FeedforwardTimeSeries, self).__init__()
        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input (batch, seq_length, input_size) -> (batch, seq_length * input_size)
        x = x.view(x.size(0), -1)
        out = self.network(x)
        return out