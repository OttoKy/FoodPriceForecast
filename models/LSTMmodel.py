import torch
import torch.nn as nn
import torch.optim as optim


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer with the specified input size, hidden size, and number of layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define the fully connected (linear) layer that maps the LSTM's output to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)
        
    
    
    def forward(self, x):
        # Initialize the hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Pass the input data through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Select the last hidden state from the LSTM layer's output
        out = self.fc(out[:, -1, :])
        return out
    
    
    
    
    

