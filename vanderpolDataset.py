import pandas as pd
import torch

class VDPDataset:
    def __init__(self, csv_path,normalize=True):
        # Load the data from the CSV file
        data = pd.read_csv(csv_path)

        # Extract the input and output columns
        self.u = torch.tensor(data['u'].values, dtype=torch.float32).view(-1, 1)  # Ensure u is a column vector
        self.y = torch.tensor(data['y'].values, dtype=torch.float32).view(-1, 1)  # Ensure y is a column vector

        self.dt = data['time'][1] - data['time'][0]

        self.warmup_window = 0
        self.x1 = torch.tensor(data['x1'].values, dtype=torch.float32).view(-1, 1)
        self.x2 = torch.tensor(data['x2'].values, dtype=torch.float32).view(-1, 1)

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

        self.u_plot = self.u_test[1:]
        self.y_plot = self.y_test[1:]
        x1_plot = self.y_test[1,0]
        x2_plot = (self.y_test[2] - self.y_test[0]) / (2 * self.dt)
        self.starting_state_plot = torch.tensor([[x1_plot, x2_plot]], dtype=torch.float32)