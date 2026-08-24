import torch
import torch.nn as nn

class Interpretable2DModel(nn.Module):
    def __init__(self,dt, device='cpu'):
        super(Interpretable2DModel, self).__init__()
        # Define the interpretable model architecture here
        # For example, a simple linear model
        self.device = device
        self.dt = dt

        self.a21 = nn.Parameter(torch.tensor(-3.0982e+02, device=self.device, dtype=torch.double), requires_grad=True)
        self.a22 = nn.Parameter(torch.tensor(4.1921e-01, device=self.device, dtype=torch.double), requires_grad=True)
        self.b2  = nn.Parameter(torch.tensor(104.0715, device=self.device, dtype=torch.double), requires_grad=True)

        self.register_buffer('C', torch.tensor([[1.0, 0.0]], device=self.device, dtype=torch.double))
        self.register_buffer('D', torch.tensor([[0.0]], device=self.device, dtype=torch.double))
        self.register_buffer("A", torch.tensor([[1.0, 0.0], [-3.5497e+01, 8.4196e-01]], device=self.device, dtype=torch.double))
        self.register_buffer("B", torch.tensor([[0.0], [-2.0555]], device=self.device, dtype=torch.double))





    def forward(self, state, u):
        """
        Forward pass for the interpretable model.
        
        Args:
            state (Tensor): Current state of the system [batch_size, state_dim].
            u (Tensor): Control input [batch_size, input_dim].
        Returns:
            next_state (Tensor): Predicted next state [batch_size, state_dim].
            y (Tensor): Predicted output [batch_size, output_dim].
        """
        A = torch.stack([
            torch.stack([torch.tensor(1.0, device=self.device, dtype=torch.double), torch.tensor(self.dt, device=self.device, dtype=torch.double)]),
            torch.stack([self.a21, self.a22]),
        ])
        B = torch.stack([
            torch.stack([torch.tensor(0.0, device=self.device, dtype=torch.double)]),
            torch.stack([self.b2]),
        ])
        self.A.copy_(A.detach())
        self.B.copy_(B.detach())

        next_state = state @ A.T + u @ B.T
        next_y = state @ self.C.T + u @ self.D.T
        return next_state, next_y
