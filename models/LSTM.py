import torch
import torch.nn as nn

class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMmodel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define the fully connected (linear) layer that maps the LSTM's output to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)

        # Define the loss function
        self.loss_function = nn.MSELoss()

    
    def forward(self, x):
        # Initialize the hidden and cell states with zeros
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device, dtype=x.dtype)


        # Pass input through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Get the last hidden state from LSTM output
        out = self.fc(out[:, -1, :])
        return out



    
    
    
    
    

