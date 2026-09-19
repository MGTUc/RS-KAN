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
            # Dimensionless time: x2 := dt * dy/dt, i.e. the per-sample increment of y,
            # so A[0,1] = 1 instead of dt. With physical dt the silverbox resonance
            # (69 Hz) forces std(x2)/std(x1) ~ 390 and a21 ~ -288, which neither Adam
            # nor the KAN can reach from an O(1) initialisation. In this basis the
            # optimum is a21=-0.473, a22=0.462, b2=0.166 and both states are O(1).
            self.h = 1.0
            self.a21 = nn.Parameter(torch.tensor(0.0, device=self.device, dtype=torch.float32), requires_grad=True)
            self.a22 = nn.Parameter(torch.tensor(1.0, device=self.device, dtype=torch.float32), requires_grad=True)
            self.b2  = nn.Parameter(torch.tensor(0.0, device=self.device, dtype=torch.float32), requires_grad=True)

            self.register_buffer('C', torch.tensor([[1.0, 0.0]], device=self.device, dtype=torch.float32))
            self.register_buffer('D', torch.tensor([[0.0]], device=self.device, dtype=torch.float32))
            self.register_buffer("A", torch.tensor([[1.0, self.h], [0.0, 0.9]], device=self.device, dtype=torch.float32))
            self.register_buffer("B", torch.tensor([[0.0], [0.0]], device=self.device, dtype=torch.float32))
        elif case_name == "vdp":
            self.h = self.dt  # vdp is already well-scaled at physical dt (std(x2)/std(x1) ~ 1)
            self.a21 = nn.Parameter(torch.tensor(0, device=self.device, dtype=torch.float32), requires_grad=True)
            self.a22 = nn.Parameter(torch.tensor(1, device=self.device, dtype=torch.float32), requires_grad=True)
            self.b2  = nn.Parameter(torch.tensor(0, device=self.device, dtype=torch.float32), requires_grad=True)

            self.register_buffer('C', torch.tensor([[1.0, 0.0]], device=self.device, dtype=torch.float32))
            self.register_buffer('D', torch.tensor([[0.0]], device=self.device, dtype=torch.float32))
            self.register_buffer("A", torch.tensor([[1.0, 0.025], [-0.025, 1.01]], device=self.device, dtype=torch.float32))
            self.register_buffer("B", torch.tensor([[0.0], [0.025]], device=self.device, dtype=torch.float32))





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
                torch.stack([torch.tensor(1.0, device=self.device, dtype=torch.float32), torch.tensor(self.h, device=self.device, dtype=torch.float32)]),
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

    def plot(self, u, starting_state, warmup_window=0, **kwargs):
        """
        Plots the interpretable model's dynamics.

        Args:
            u (Tensor): Control input, [seq_len, input_dim] for a single trajectory or
                [seq_len, batch, input_dim] for several trajectories rolled out in
                parallel (one per row of starting_state).
            starting_state (Tensor): Initial state(s) [batch, state_dim].
            warmup_window (int): Number of initial time steps to exclude from plotting.
            **kwargs: Additional keyword arguments for the plot function.
        """
        batched = u.dim() == 3

        with torch.no_grad():
            current_state = starting_state.clone()
            state_list = []
            state_list.append(current_state.clone())
            for t in range(u.size(0)):
                current_input = u[t] if batched else u[t].unsqueeze(0)
                next_state, _ = self.forward(current_state, current_input)
                state_list.append(next_state.clone())
                current_state = next_state
            state = torch.stack(state_list[1:], dim=0) if batched else torch.cat(state_list[1:])

        if batched:
            # Flatten [seq_len, batch, dim] -> [seq_len*batch, dim], pairing each
            # trajectory's own rolled-out state with its own input at each step.
            state_for_activations = state[warmup_window:].reshape(-1, state.shape[-1])
            u_for_activations = u[warmup_window:].reshape(-1, u.shape[-1])
        else:
            state_for_activations = state[warmup_window:]
            u_for_activations = u[warmup_window:]
        self.forward(state_for_activations, u_for_activations)  # Forward pass to save activations for plotting

        if self.state_kan is not None:
            self.state_kan.plot(**kwargs)
        else:
            print("No state_kan module available for plotting.")
