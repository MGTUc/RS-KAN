import torch
import torch.nn as nn
from rskan import RSKAN

class FullStateNonlinearityRSKAN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, zero_final_layer=True, **kan_kwargs):
        """
        Initializes the KAN-based nonlinearity module.
        
        Args:
            input_size (int): Total number of input features (state_dim + control input if used).
            hidden_layers (int): Hidden layer size (or can be a list of layer sizes).
            output_size (int): Output dimension (should match the state correction dimension).
            zero_final_layer (bool): If True, the final layer weights are initialized to zero.
            **kan_kwargs: Additional keyword arguments for rskan.RSKAN.
        """
        super(FullStateNonlinearityRSKAN, self).__init__()
        # Ensure layers_hidden is a list for KAN constructor
        if isinstance(hidden_layers, int):
             layers_config = [input_size, hidden_layers, output_size]
        else: # Assuming hidden_layers is already a list/tuple
             layers_config = [input_size] + list(hidden_layers) + [output_size]
        self.kan = RSKAN(layers_config, **kan_kwargs)

        if zero_final_layer:
            with torch.no_grad():
                final_layer = self.kan.layers[-1]
                final_layer.residual_scaling.zero_()
                final_layer.subnet_scaling.zero_()

    def forward(self, state=None, u=None, v=None, update_grid=False):
        """
        Forward pass for the KAN nonlinearity.
        
        Args:
            v (torch.Tensor, optional): Intermediate output. If provided, the model processes v.
            x (torch.Tensor, optional): State. Must be provided with u if v is None.
            u (torch.Tensor, optional): Input. Must be provided with x if v is None.
            update_grid : this is not used but is kept for compatibility with the SSmodel interface. It is ignored in this implementation.
        Returns:
            Tensor: Nonlinear correction.
        """
        if v is not None:
            # Process intermediate output v
            inp = v
        elif state is not None and u is not None:            
            inp = torch.cat([state, u], dim=-1)
        elif state is not None and u is None:            
            inp = state

        return self.kan(inp, save_activations=True)
    
    def plot(self,u, **kwargs):
        """
        Plots the KAN nonlinearity.
        
        Args:
            u (torch.Tensor): Input sequence for simulating the state evolution.
            **kwargs: Additional keyword arguments for the plot function.
        """
        self.kan.eval()
        
        input_dim = self.kan.layers[0].input_size
        u_dim = u.shape[1] if u is not None else 0

        state_dim = input_dim - u_dim

        with torch.no_grad():
            print(f"    Simulating full train sequence ({len(u)} steps)...")
            current_state = torch.zeros(1, state_dim)
            state_list = []
            state_list.append(current_state.clone()) # Store x(0) guess
            for t in range(len(u)):
                current_input = u[t].unsqueeze(0)
                next_state = self.forward(current_state, current_input)
                state_list.append(next_state.clone()) # Store x(t+1)
                current_state = next_state
            state = torch.cat(state_list[1:])
        self.forward(state, u)
        self.kan.plot(**kwargs)