import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class FoodPriceDataset(Dataset):
    

    def __init__(self, data, sequence_length, target_col=None, normalize=True):
        
        #Initializes the dataset
        
        self.sequence_length = sequence_length
        self.target_col = target_col if target_col is not None else -1
        self.normalize = normalize
        if self.normalize:
            data, self.scaler = self._normalize_data(data)
        
        self.data = data

        
        
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def _normalize_data(self, data):
        
        # Normalizes the data
        
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        return data, scaler
    
    
    def __getitem__(self,idx):
        
        # returns sequence of input data and target value
        
        x = self.data[idx:idx+self.sequence_length, :]
        y = self.data[idx+self.sequence_length, self.target_col]
        return x, y
    

class TimeSeriesDataLoader(DataLoader):
    # A wrapper class for the PyTorch DataLoader class
    
    def __init__(self, data, sequence_length, target_cols=None, batch_size=32, normalize=True):
        dataset = FoodPriceDataset(data, sequence_length, target_cols, normalize=True)
        super().__init__(dataset, batch_size=batch_size)
        self.num_features = dataset.data.shape[1]
        



