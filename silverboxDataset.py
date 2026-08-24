import nonlinear_benchmarks
import torch

class SilverboxDataset():
    def __init__(self, normalize=True):
        train_val, test = nonlinear_benchmarks.Silverbox(atleast_2d=True)
        self.dt = train_val.sampling_time
        

        self.u, self.y = torch.tensor(train_val.u), torch.tensor(train_val.y)
        test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
        self.u_test, self.y_test = torch.tensor(test_arrow_no_extrapolation.u), torch.tensor(test_arrow_no_extrapolation.y)

        self.warmup_window = test_arrow_no_extrapolation.state_initialization_window_length

        if normalize:
            self.u_mean, self.u_std = self.u.mean(), self.u.std(dim=0)
            self.y_mean, self.y_std = self.y.mean(dim=0), self.y.std(dim=0)

            self.u_train = (self.u - self.u_mean) / self.u_std
            self.y_train = (self.y - self.y_mean) / self.y_std

            self.u_test = (self.u_test - self.u_mean) / self.u_std
            self.y_test = (self.y_test - self.y_mean) / self.y_std
        else:
            self.u_mean, self.u_std = 0.0, 1.0
            self.y_mean, self.y_std = 0.0, 1.0
            self.u_train = self.u
            self.y_train = self.y

            self.u_test = self.u_test
            self.y_test = self.y_test
