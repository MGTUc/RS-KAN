import pandas as pd
import torch

class VDPDataset:
    def __init__(self, normalize=True):
        # Load the data from the CSV file
        data = pd.read_csv('data/vanderpolData.csv')

        # Extract the input and output columns
        self.u = torch.tensor(data['u'].values, dtype=torch.float32).view(-1, 1)  # Ensure u is a column vector
        self.y = torch.tensor(data['y'].values, dtype=torch.float32).view(-1, 1)  # Ensure y is a column vector

        self.dt = data['time'][1] - data['time'][0]

        self.warmup_window = 0  # Set to 0 for the Vanderpol dataset, as there is no warmup window

        if normalize:
            self.u_mean, self.u_std = self.u.mean(), self.u.std()
            self.y_mean, self.y_std = self.y.mean(), self.y.std()

            self.u_train = (self.u - self.u_mean) / self.u_std
            self.y_train = (self.y - self.y_mean) / self.y_std

            self.u_test = self.u_train.clone()
            self.y_test = self.y_train.clone()
        else:
            self.u_mean, self.u_std = 0.0, 1.0
            self.y_mean, self.y_std = 0.0, 1.0
            self.u_train = self.u
            self.y_train = self.y

            self.u_test = self.u.clone()
            self.y_test = self.y.clone()