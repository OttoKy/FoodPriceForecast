import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from models.LSTM import LSTMmodel
from data.data_loader import TimeSeriesDataLoader
from utils.metrics import mean_absolute_error, root_mean_squared_error
from utils.config import learning_rate, num_epochs, input_size, hidden_size, num_layers, output_size

# Device and data type configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64

# Load data and bit more preprocesseing
df = pd.read_csv('data/processed/cleaned.csv')
df = df.drop("Date", axis=1)
data = df.to_numpy()

# Set parameters for data loader
sequence_length = 10
batch_size = 32
target_col = [2, 3, 4, 5, 6]
input_cols = [0,1]

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
train_data, val_data = train_test_split(train_data, test_size=0.2, shuffle=False)

# Create data loaders
train_loader = TimeSeriesDataLoader(train_data, sequence_length, input_cols=input_cols, target_col=target_col, batch_size=batch_size, normalize=True)
val_loader = TimeSeriesDataLoader(val_data, sequence_length, input_cols=input_cols, target_col=target_col, batch_size=batch_size, normalize=True)
test_loader = TimeSeriesDataLoader(test_data, sequence_length, input_cols=input_cols, target_col=target_col, batch_size=batch_size, normalize=True)

input_size = len(input_cols)

# Initialize model
model = LSTMmodel(input_size, hidden_size, num_layers, output_size).to(device, dtype=dtype)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_batches = 0
    train_mae, train_rmse = [], []
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(torch.float64).to(device)
        targets = targets.to(torch.float64).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_batches += 1
        mae = mean_absolute_error(outputs, targets).detach().cpu().numpy()
        rmse = root_mean_squared_error(outputs, targets).detach().cpu().numpy()
        train_mae.append(mae)
        train_rmse.append(rmse)

    train_loss /= train_batches
    train_mae = np.mean(train_mae)
    train_rmse = np.mean(train_rmse)

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_batches = 0
        val_mae, val_rmse = [], []
        for inputs, targets in val_loader:
            inputs = inputs.to(torch.float64).to(device)
            targets = targets.to(torch.float64).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_batches += 1

            mae = mean_absolute_error(outputs, targets).detach().cpu().numpy()
            rmse = root_mean_squared_error(outputs, targets).detach().cpu().numpy()
            val_mae.append(mae)
            val_rmse.append(rmse)

        val_loss /= val_batches
        val_mae = np.mean(val_mae)
        val_rmse = np.mean(val_rmse)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}, Validation RMSE: {val_rmse:.4f}")


# 
actual_predictions = outputs.detach().cpu().numpy()
actual_predictions_denormalized = test_loader.dataset.denormalize_targets(actual_predictions)

print(actual_predictions)
print(actual_predictions_denormalized)