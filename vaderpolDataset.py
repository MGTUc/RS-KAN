import pandas as pd
import torch

class VDPDataset:
    def __init__(self, normalize=True):
        # Load the data from the CSV file
        data = pd.read_csv('data/vanderpolDataFree.csv')

        # Extract the input and output columns
        self.u = torch.tensor(data['u'].values, dtype=torch.float32).view(-1, 1)  # Ensure u is a column vector
        self.y = torch.tensor(data['y'].values, dtype=torch.float32).view(-1, 1)  # Ensure y is a column vector

        self.dt = data['time'][1] - data['time'][0]

        self.warmup_window = 50

        x1 = torch.tensor(data['x1'].values, dtype=torch.float32).view(-1, 1)
        x2 = torch.tensor(data['x2'].values, dtype=torch.float32).view(-1, 1)
        print(f"x1 mean: {x1.mean()}, x1 std: {x1.std()}")
        print(f"x2 mean: {x2.mean()}, x2 std: {x2.std()}")

        if normalize:
            self.u_mean, self.u_std = self.u[:-1500,0].mean(), self.u.std()
            self.y_mean, self.y_std = self.y[:-1500,0].mean(), self.y.std()

            self.u_train = (self.u - self.u_mean) / self.u_std
            self.y_train = (self.y - self.y_mean) / self.y_std

            self.u_test = (self.u[-1500:] - self.u_mean) / self.u_std
            self.y_test = (self.y[-1500:] - self.y_mean) / self.y_std

            print(f"u_mean: {self.u_mean}, u_std: {self.u_std}")
            print(f"y_mean: {self.y_mean}, y_std: {self.y_std}")
        else:
            self.u_mean, self.u_std = 0.0, 1.0
            self.y_mean, self.y_std = 0.0, 1.0
            self.u_train = self.u[:-1500]
            self.y_train = self.y[:-1500]

            self.u_test = self.u[-1500:]
            self.y_test = self.y[-1500:]