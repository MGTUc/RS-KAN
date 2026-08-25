import torch
import torch.nn as nn

class Interpretable2DModel(nn.Module):
    def __init__(self,dt, case_name,linear_trainable = True, state_kan = None, device='cpu'):
        super(Interpretable2DModel, self).__init__()
        # Define the interpretable model architecture here
        # For example, a simple linear model
        self.device = device
        self.dt = dt
        self.linear_trainable = linear_trainable
        self.state_kan = state_kan

        if case_name == "silverbox":
            self.a21 = nn.Parameter(torch.tensor(-3.0982e+02, device=self.device, dtype=torch.float32), requires_grad=True)
            self.a22 = nn.Parameter(torch.tensor(4.1921e-01, device=self.device, dtype=torch.float32), requires_grad=True)
            self.b2  = nn.Parameter(torch.tensor(104.0715, device=self.device, dtype=torch.float32), requires_grad=True)

            self.register_buffer('C', torch.tensor([[1.0, 0.0]], device=self.device, dtype=torch.float32))
            self.register_buffer('D', torch.tensor([[0.0]], device=self.device, dtype=torch.float32))
            self.register_buffer("A", torch.tensor([[1.0, 0.0], [-3.0982e+02, 4.1921e-01]], device=self.device, dtype=torch.float32))
            self.register_buffer("B", torch.tensor([[0.0], [104.0715]], device=self.device, dtype=torch.float32))
        elif case_name == "vdp":
            self.a21 = nn.Parameter(torch.tensor(-1.3, device=self.device, dtype=torch.float32), requires_grad=True)
            self.a22 = nn.Parameter(torch.tensor(0.3, device=self.device, dtype=torch.float32), requires_grad=True)
            self.b2  = nn.Parameter(torch.tensor(0.3, device=self.device, dtype=torch.float32), requires_grad=True)

            self.register_buffer('C', torch.tensor([[1.0, 0.0]], device=self.device, dtype=torch.float32))
            self.register_buffer('D', torch.tensor([[0.0]], device=self.device, dtype=torch.float32))
            self.register_buffer("A", torch.tensor([[1.0, 0.0], [-1.3, 0.3]], device=self.device, dtype=torch.float32))
            self.register_buffer("B", torch.tensor([[0.0], [0.4]], device=self.device, dtype=torch.float32))





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
        if self.linear_trainable:
            A = torch.stack([
                torch.stack([torch.tensor(1.0, device=self.device, dtype=torch.float32), torch.tensor(self.dt, device=self.device, dtype=torch.float32)]),
                torch.stack([self.a21, self.a22]),
            ])
            B = torch.stack([
                torch.stack([torch.tensor(0.0, device=self.device, dtype=torch.float32)]),
                torch.stack([self.b2]),
            ])
            self.A.copy_(A.detach())
            self.B.copy_(B.detach())
        else:
            A = self.A
            B = self.B

        next_state = state @ A.T + u @ B.T
        next_y = state @ self.C.T + u @ self.D.T

        if self.state_kan is not None:
            # Apply the KAN nonlinearity to the state
            next_state += self.state_kan(state=state, u=u)

        return next_state, next_y

    def plot(self, u, warmup_window=0, **kwargs):
        """
        Plots the interpretable model's dynamics.
        
        Args:
            u (Tensor): Control input [seq_len, input_dim].
            warmup_window (int): Number of initial time steps to exclude from plotting.
            **kwargs: Additional keyword arguments for the plot function.
        """

        with torch.no_grad():
            current_state = torch.zeros(1, 2, dtype=torch.float32)
            state_list = []
            state_list.append(current_state.clone()) 
            for t in range(len(u)):
                current_input = u[t].unsqueeze(0)
                next_state, _ = self.forward(current_state, current_input)
                state_list.append(next_state.clone())
                current_state = next_state
            state = torch.cat(state_list[1:])
        self.forward(state[warmup_window:], u[warmup_window:])  # Forward pass to save activations for plotting

        if self.state_kan is not None:
            self.state_kan.plot(**kwargs)
        else:
            print("No state_kan module available for plotting.")
