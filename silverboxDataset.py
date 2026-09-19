import nonlinear_benchmarks
import torch

class SilverboxDataset():
    def __init__(self, normalize=True):
        train_val, test = nonlinear_benchmarks.Silverbox(atleast_2d=True)
        self.dt = train_val.sampling_time
        

        self.u, self.y = torch.tensor(train_val.u, dtype=torch.float32), torch.tensor(train_val.y, dtype=torch.float32)
        test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
        self.u_test, self.y_test = torch.tensor(test_multisine.u, dtype=torch.float32), torch.tensor(test_multisine.y, dtype=torch.float32)

        self.warmup_window = test_multisine.state_initialization_window_length
        print(f"Silverbox dataset: warmup window length = {self.warmup_window}")

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

        self.u_plot = self.u_test[1:]
        self.y_plot = self.y_test[1:]
        x1_plot = self.y_test[1, 0]
        x2_plot = (self.y_test[2, 0] - self.y_test[0, 0]) / 2  # dimensionless time: x2 := dt*dy/dt
        self.starting_state_plot = torch.tensor([[x1_plot, x2_plot]], dtype=torch.float32)

    def linear_init(self):
        """Least-squares initialisation for (a21, a22, b2) in dimensionless time.

        Fits an ARX(2,2) model y[t+1] = a1 y[t] + a2 y[t-1] + b0 u[t] + b1 u[t-1] to the
        training data, then maps it onto x1 = y, x2 = y[t+1] - y[t]:
            a21 = (a1 - 1) + a2,   a22 = a1 - 1,   b2 = b0
        Free-run test RMSE of this linear model alone is ~0.14 (y_test std ~1.0), so the
        KAN starts from a correct linear model and only has to learn the ~1.7% residual.
        """
        y = self.y_train[:, 0]
        u = self.u_train[:, 0]
        T = len(y) - 2
        M = torch.stack([y[1:1 + T], y[0:T], u[1:1 + T], u[0:T]], dim=1)
        a1, a2, b0, b1 = torch.linalg.lstsq(M, y[2:2 + T].unsqueeze(1)).solution[:, 0].tolist()
        return (a1 - 1) + a2, a1 - 1, b0
