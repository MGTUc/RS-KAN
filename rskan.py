import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os

class RSKANlayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        subnetwork_hidden_shape=[10, 10],
        input_layer=False,
        residual_connection=False,
        subnet_scaling=1.0,
        residual_scaling=1.0,
        mask=None
    ):
        """
        A single RSKAN layer: each (input, output) pair has its own small MLP subnetwork.

        Args:
            input_size (int): Number of input nodes to this layer.
            output_size (int): Number of output nodes from this layer.
            subnetwork_hidden_shape (list[int]): Hidden layer widths of each edge subnetwork.
            input_layer (bool): Whether this is the first layer in the network. Affects how
                the residual scaling is initialised.
            residual_connection (bool): If True, adds a learnable scaled residual path that
                bypasses the subnetwork. The subnet_scaling is then initialised to zero so
                training starts from the residual branch.
            subnet_scaling (float): Initial scale factor applied to the subnetwork output.
                Ignored when residual_connection=True (subnet starts at 0 in that case).
            residual_scaling (float): Initial scale factor applied to the residual (identity)
                path. Only used when residual_connection=True.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_nets = input_size * output_size
        self.pre_activations = None
        self.post_activations = None
        
        full_shape = [1] + subnetwork_hidden_shape + [1]


        # Mask shape: [input_size, output_size]
        if mask is None:
            mask = torch.ones(input_size, output_size, dtype=torch.float32)
        else:
            mask = torch.as_tensor(mask, dtype=torch.float32)
            assert mask.shape == (input_size, output_size), \
                f"Mask shape {mask.shape} must match (input_size, output_size) = ({input_size}, {output_size})"

        self.register_buffer('mask', mask)
        self.register_buffer('mask_flat', self.mask.view(self.num_nets, 1, 1))

        
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        for i in range(len(full_shape) - 1):
            in_dim = full_shape[i]
            out_dim = full_shape[i+1]
            
            # Weight shape: [Num_Nets, Out_Dim, In_Dim]
            # Bias shape:  [Num_Nets, Out_Dim, 1]
            lastLayer = i == len(full_shape) - 2

            w = nn.Parameter(self._init_scaled_weights(self.num_nets, out_dim, in_dim, lastLayer))
            b = nn.Parameter(torch.zeros(self.num_nets, out_dim, 1, dtype=torch.float32))
            
            self.weights.append(w)
            self.biases.append(b)
        
        masked_inputs_per_output = self.mask.sum(dim=0, keepdim=True) # [1, output_size]
        if residual_connection:
            
            if input_layer:
                scaling_factor = residual_scaling / torch.sqrt(masked_inputs_per_output + 1e-8) # [1, output_size]
                self.residual_scaling = nn.Parameter(
                    (scaling_factor.expand(input_size, output_size) * self.mask).view(self.num_nets, 1, 1)
                )
                # self.residual_scaling = nn.Parameter(
                #     torch.ones(self.num_nets, 1, 1) * (residual_scaling / np.sqrt(input_size))
                # )
            else:
                scaling_factor = residual_scaling / (masked_inputs_per_output + 1e-8) # [1, output_size]

                self.residual_scaling = nn.Parameter(
                    (scaling_factor.expand(input_size, output_size) * self.mask).view(self.num_nets, 1, 1)
                )
                # self.residual_scaling = nn.Parameter(
                #     torch.ones(self.num_nets, 1, 1) * (residual_scaling / np.sqrt(input_size + input_size * (input_size-1)))
                # )
            self.subnet_scaling = nn.Parameter(
                torch.zeros(self.num_nets, 1, 1, dtype=torch.float32)
            )
        else:
            self.register_buffer('residual_scaling', torch.zeros(self.num_nets, 1, 1, dtype=torch.float32))
            
            scaling_factor = subnet_scaling / torch.sqrt(masked_inputs_per_output + 1e-8) # [1, output_size]
            # and then expand to [input_size, output_size] and flatten to [num_nets, 1, 1]
            self.subnet_scaling = nn.Parameter(
                (scaling_factor.expand(input_size, output_size) * self.mask).view(self.num_nets, 1, 1)
            )

            # self.subnet_scaling = nn.Parameter(
            #     torch.ones(self.num_nets, 1, 1) * (subnet_scaling / np.sqrt(self.input_size))
            # )
            


    @staticmethod
    def _init_scaled_weights(num_nets, out_dim, in_dim, lastLayer):
        """
        Initialises weights for a batch of subnetwork layers using He-style scaling.

        Args:
            num_nets (int): Number of subnetworks (= input_size * output_size).
            out_dim (int): Output width of this weight layer.
            in_dim (int): Input width of this weight layer.
            lastLayer (bool): If True, uses 1/sqrt(in_dim) std instead of the
                He gain, since the final subnetwork layer does not have a non-linearity.

        Returns:
            torch.Tensor: Weight tensor of shape [num_nets, out_dim, in_dim].
        """
        gain = np.sqrt(2.0)
        std = (gain / np.sqrt(in_dim)) if not lastLayer else (1.0 / np.sqrt(in_dim))
        weights = torch.randn(num_nets, out_dim, in_dim, dtype=torch.float32) * std
 
        return weights
    
    def forward(self, x, save_activations=True):
        """
        Forward pass through all edge subnetworks, then sums contributions per output node.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_size].
            save_activations (bool): If True, stores pre- and post-activation tensors on the
                layer object so they can be retrieved for plotting or regularisation.
                Pre-activations have shape [batch_size, input_size]; post-activations have
                shape [input_size, output_size, batch_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_size].
        """
        if save_activations:
            # Keep graph-connected activations so regularization contributes gradients.
            self.pre_activations = x
        batch_size = x.shape[0]
        
        # Result shape: [In*Out N, 1, Batch]
        x = x.T.repeat_interleave(self.output_size, dim=0).unsqueeze(1)
        x_init = x
        
        num_layers = len(self.weights)
        for i in range(num_layers):
            # Weight [N, Out, In] @ Input [N, In, Batch] -> [N, Out, Batch]
            x = torch.matmul(self.weights[i], x) + self.biases[i]
            
            if i < num_layers - 1:
                x = torch.nn.functional.silu(x)


        # x is currently [Num_Nets, 1, Batch] because the last layer output_dim is 1
        x = x * self.subnet_scaling + x_init * self.residual_scaling

        x = x * self.mask_flat

        x = x.view(self.input_size, self.output_size, batch_size)
        if save_activations:
            self.post_activations = x
        # Sum over input_size (dim 0) -> [output_size, batch_size] -> Transpose to [Batch, Out]
        return (x.sum(dim=0)).T 
    



class RSKAN(nn.Module):
    def __init__(
        self,
        layerSizes,
        subnetwork_shape=[10,10],
        residual_connection=False,
        subnet_scaling=1.0,
        residual_scaling=1.0,
        masks=None
    ):
        """
        Residual Subnetwork KAN: a stack of RSKANlayers forming a full network.

        Args:
            layerSizes (list[int]): Width of each layer, including input and output.
                E.g. [4, 8, 2] creates a network with 4 inputs, one hidden layer of
                width 8, and 2 outputs.
            subnetwork_shape (list[int]): Hidden layer widths shared by every edge
                subnetwork across all layers.
            residual_connection (bool): If True, each RSKANlayer gets a learnable
                residual path (see RSKANlayer).
            subnet_scaling (float): Initial output scale for the subnetwork branch.
                Passed to each RSKANlayer.
            residual_scaling (float): Initial output scale for the residual branch.
                Only relevant when residual_connection=True.
        """
        super(RSKAN, self).__init__()
        self.subnetwork_shape = subnetwork_shape

        self.layerSizes = layerSizes
        self.edge_attribution_scores = [[[1.0 for _ in range(layerSizes[l+1])] for _ in range(layerSizes[l])] for l in range(len(layerSizes)-1)]
        self.node_attribution_scores = [[1.0 for _ in range(layerSizes[i])] for i in range(len(layerSizes))]
        layers = []

        for i in range(len(layerSizes)-1):
            layer_mask = masks[i] if masks is not None else None
            rskan_layer = RSKANlayer(
                input_size=layerSizes[i],
                output_size=layerSizes[i+1],
                subnetwork_hidden_shape=self.subnetwork_shape,
                input_layer=(i == 0),
                subnet_scaling=subnet_scaling,
                residual_scaling=residual_scaling,
                residual_connection=residual_connection,
                mask=layer_mask,
            )
            layers.append(rskan_layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x, save_activations=True):
        """
        Forward pass through all RSKAN layers sequentially.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, layerSizes[0]].
            save_activations (bool): If True, each layer stores its pre- and
                post-activations. Required before calling plot() or regularization_loss().

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, layerSizes[-1]].
        """
        if save_activations:
            for layer in self.layers:
                x = layer(x, save_activations=save_activations)
        else:
            x = self.layers(x)
        return x
    
    def regularization_loss(self, reg_activation = 1, reg_entropy = 1):
        """
        Computes sparsity regularisation over saved post-activations.

        Requires a prior forward pass with save_activations=True.

        Args:
            reg_activation (float): Weight for the L1 activation magnitude penalty.
                Set to 0 to disable.
            reg_entropy (float): Weight for the entropy-based sparsity penalty, which
                encourages each edge to specialise on a small subset of inputs/outputs.
                Set to 0 to disable.

        Returns:
            torch.Tensor: Scalar regularisation loss.
        """
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            if not isinstance(layer, RSKANlayer):
                continue

            if reg_activation != 0 and layer.post_activations is not None:
                reg_loss += reg_activation * torch.mean(torch.abs(layer.post_activations))

            if reg_entropy != 0 and layer.post_activations is not None:
                # activations = torch.abs(layer.post_activations)
                activations = torch.mean(torch.abs(layer.post_activations), dim=2)
                p_input = torch.div(activations, activations.sum(dim=0, keepdim=True) + 1e-8)
                p_output = torch.div(activations, activations.sum(dim=1, keepdim=True) + 1e-8)
                input_entropy = -torch.sum(p_input * torch.log(p_input + 1e-8), dim=0).mean()
                output_entropy = -torch.sum(p_output * torch.log(p_output + 1e-8), dim=1).mean()
                reg_loss += reg_entropy * (input_entropy + output_entropy)
        return reg_loss
    
    def get_activation(self, layer_idx, input_idx, output_idx):
        """
        Returns the saved pre- and post-activations for a specific edge.

        Requires a prior forward pass with save_activations=True.

        Args:
            layer_idx (int): Index of the layer.
            input_idx (int): Index of the input node at that layer.
            output_idx (int): Index of the output node at that layer.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - pre: tensor of shape [batch_size] with input values for input_idx.
                - post: tensor of shape [batch_size] with the subnetwork output for the
                  (input_idx, output_idx) edge.

        Raises:
            ValueError: If no activations are stored for the requested layer.
        """
        layer = self.layers[layer_idx]
        if isinstance(layer, RSKANlayer) and layer.post_activations is not None:
            return layer.pre_activations[:, input_idx], layer.post_activations[input_idx, output_idx, :]
        else:
            raise ValueError(f"No activations found for layer {layer_idx}. Run forward pass with save_activations=True.")
    
    def plot(self, folder="./figures", save_final_figure = True, attribution_score_alpha=True, scale=0.5, tick=True, sample=False, in_vars=None, out_vars=None, title=None, edge_plot_scale=1.5):
        """
        Plot an RSKAN architecture with per-edge activation function thumbnails.

        Requires a prior forward pass with save_activations=True.

        Args:
            folder (str): Directory where edge thumbnail images and the final figure are
                saved. Created automatically if it does not exist.
            save_final_figure (bool): If True, saves the assembled network diagram to
                ``<folder>/RSKAN.pdf``.
            attribution_score_alpha (bool): If True, computes per-edge attribution scores
                and uses them to set line and plot alpha (low-contribution edges fade out).
            scale (float): Global size multiplier for the figure and node markers.
            tick (bool): If True, draws axis ticks on each edge thumbnail. to False.
            sample (bool): If True, overlays scatter points on each edge thumbnail in
                addition to the line plot.
            in_vars (list[str] | None): Labels for the input nodes. If None, no labels are
                drawn.
            out_vars (list[str] | None): Labels for the output nodes. If None, no labels
                are drawn.
            title (str | None): Title displayed above the network diagram. If None, no
                title is shown.
            edge_plot_scale (float): Additional size multiplier applied only to the edge
                thumbnail images.
        """
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        print(self.layers[0].pre_activations.var(dim=0))

        if attribution_score_alpha:
            for l in reversed(range(len(self.layers))):
                for i in range(self.layerSizes[l]):
                    for j in range(self.layerSizes[l + 1]):
                        self.edge_attribution_scores[l][i][j] = self.node_attribution_scores[l+1][j] * (torch.std(self.layers[l].post_activations[i, j, :])/(torch.std((self.layers[l].post_activations[:,j,:]).sum(dim=0))) + 1e-8).item()
                    self.node_attribution_scores[l][i] = sum(self.edge_attribution_scores[l][i][j] for j in range(self.layerSizes[l + 1]))
                    

        if attribution_score_alpha:
            max_node_score = max(max(layer) for layer in self.node_attribution_scores) + 1e-8
            max_edge_score = max(max(max(out_scores) for out_scores in layer) for layer in self.edge_attribution_scores) + 1e-8

            self.node_attribution_scores = [[max(0, min(score/max_node_score, 1.0)) for score in layer_scores] for layer_scores in self.node_attribution_scores]
            self.edge_attribution_scores = [[[max(0, min(score/max_edge_score, 1.0)) for score in out_scores] for out_scores in layer_scores] for layer_scores in self.edge_attribution_scores]

        missing_acts = [
            idx
            for idx, layer in enumerate(self.layers)
            if layer.pre_activations is None or layer.post_activations is None
        ]
        if missing_acts:
            print(
                "No activations saved for all layers. "
                "Run model(x, save_activations=True) before calling plot()."
            )
            return None

        os.makedirs(folder, exist_ok=True)

        depth = len(self.layerSizes) - 1
        thumbnail_zoom = max(0.06, 0.16 * scale * edge_plot_scale)
        for l in range(depth):
            for i in range(self.layerSizes[l]):
                for j in range(self.layerSizes[l + 1]):
                    rank = torch.argsort(self.layers[l].pre_activations[:, i])
                    x_vals = self.layers[l].pre_activations[:, i][rank].detach().cpu().numpy()
                    y_vals = self.layers[l].post_activations[i, j, :][rank].detach().cpu().numpy()

                    edge_plot_scale = max(0.25, float(edge_plot_scale))
                    with plt.ioff():
                        fig_edge, ax_edge = plt.subplots(
                            figsize=(2.2 * edge_plot_scale, 2.2 * edge_plot_scale)
                        )
                    fig_edge.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.96)
                    if tick:
                        ax_edge.tick_params(axis="both", labelsize=8)
                    else:
                        ax_edge.set_xticks([])
                        ax_edge.set_yticks([])

                    if attribution_score_alpha:
                        edge_alpha = max(0.0, min(1.0, self.edge_attribution_scores[l][i][j]))
                        ax_edge.plot(x_vals, y_vals, color="black", lw=2.0, alpha=edge_alpha)
                    else:
                        ax_edge.plot(x_vals, y_vals, color="black", lw=2.0)
                    if sample:
                        if attribution_score_alpha:
                            edge_alpha = max(0.0, min(1.0, self.edge_attribution_scores[l][i][j]))
                            ax_edge.scatter(x_vals, y_vals, color="black", s=max(2, int(30 * scale)), alpha=edge_alpha)
                        else:
                            ax_edge.scatter(x_vals, y_vals, color="black", s=max(2, int(30 * scale)))

                    for spine in ax_edge.spines.values():
                        spine.set_color("black")
                        spine.set_linewidth(1.0)

                    fig_edge.savefig(f"{folder}/sp_{l}_{i}_{j}.png", dpi=220)
                    plt.close(fig_edge)

        width = self.layerSizes
        n_layers = len(width)
        max_nodes = max(width)

        fig_w = max(9.0, 1.8 * max_nodes) * scale * 2.0
        fig_h = max(6.0, 2.6 * (n_layers - 1)) * scale * 2.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        y_layer_pos = np.linspace(0.0, 1.0, n_layers)
        x_node_pos = []
        x_margin = 0.08
        usable_width = 1.0 - 2.0 * x_margin
        global_step = usable_width / max(max_nodes - 1, 1)
        for n in width:
            if n == 1:
                x = np.array([0.5])
            else:
                span = (n - 1) * global_step
                left = 0.5 - span / 2.0
                right = 0.5 + span / 2.0
                x = np.linspace(left, right, n)
            x_node_pos.append(x)

        for l in range(n_layers):
            for i in range(width[l]):
                ax.scatter(x_node_pos[l][i], y_layer_pos[l], s=120 * scale, color="black", zorder=5)

                if l == 0 and in_vars is not None and i < len(in_vars):
                    ax.text(
                        x_node_pos[l][i],
                        y_layer_pos[l] - 0.04,
                        str(in_vars[i]),
                        ha="center",
                        va="top",
                        fontsize=max(8, int(10 * scale)),
                    )

                if l == n_layers - 1 and out_vars is not None and i < len(out_vars):
                    ax.text(
                        x_node_pos[l][i],
                        y_layer_pos[l] + 0.04,
                        str(out_vars[i]),
                        ha="center",
                        va="bottom",
                        fontsize=max(8, int(10 * scale)),
                    )

        for l in range(n_layers - 1):
            y_bottom = y_layer_pos[l]
            y_top = y_layer_pos[l + 1]
            y_mid = 0.5 * (y_bottom + y_top)
            n_in = width[l]
            n_out = width[l + 1]
            n_edges = n_in * n_out

            if n_edges == 1:
                x_strip = np.array([0.5])
            else:
                x_strip = np.linspace(0.08, 0.92, n_edges)
            thumb_centers = {}

            for i in range(n_in):
                for j in range(n_out):
                    edge_id = i * n_out + j
                    x_img = x_strip[edge_id]
                    y_img = y_mid
                    thumb_centers[(i, j)] = (x_img, y_img)

                    img_path = f"{folder}/sp_{l}_{i}_{j}.png"
                    im = plt.imread(img_path)

                    image_box = OffsetImage(im, zoom=thumbnail_zoom)
                    ann = AnnotationBbox(image_box, (x_img, y_img), frameon=False, xycoords='data')
                    ax.add_artist(ann)

            y_gap = 0.055
            for i in range(n_in):
                for j in range(n_out):
                    x_bottom = x_node_pos[l][i]
                    x_top = x_node_pos[l + 1][j]
                    x_img, y_img = thumb_centers[(i, j)]

                    edge_alpha = self.edge_attribution_scores[l][i][j]
                    if attribution_score_alpha:
                        ax.plot([x_bottom, x_img], [y_bottom, y_img - y_gap], color="black", lw=2, alpha=edge_alpha)
                        ax.plot([x_img, x_top], [y_img + y_gap, y_top], color="black", lw=2, alpha=edge_alpha)
                    else:
                        ax.plot([x_bottom, x_img], [y_bottom, y_img - y_gap], color="black", lw=2)
                        ax.plot([x_img, x_top], [y_img + y_gap, y_top], color="black", lw=2)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.08, 1.08)
        ax.axis("off")

        if title is not None:
            ax.set_title(title, fontsize=max(10, int(12 * scale)))

        fig.tight_layout()
        if save_final_figure:
            plt.savefig(f"{folder}/RSKAN.pdf", dpi=220)
        plt.show()

# test = RSKAN(layerSizes=[3, 2, 2], 
#              subnetwork_shape=[10, 10],
#              residual_connection=True,
#              masks = [
#                 [ [0, 1],
#                   [0, 1],
#                   [0, 0] ],
#                 [ [1, 1],
#                   [1, 0]
#                 ]
#              ]
#             )
