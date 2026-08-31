import torch
from interpretable2DModel import Interpretable2DModel
from silverboxDataset import SilverboxDataset
from tqdm import tqdm
from rskanSS import FullStateNonlinearityRSKAN
from vanderpolDataset import VDPDataset
from test_bench import free_run_x1_nrmse

# torch.autograd.set_detect_anomaly(True)
seed = 1
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

epochs = 100
seq_len = 1000
learning_rate = 1e-3
enable_warmup = False
lr_warmup_epochs = 5
max_grad_norm = 1.0
stride_window = 50
batch_size = 16
report_loss_curve = True

case_name = "vdp" # silverbox or vdp

match case_name:
    case "silverbox":
        dataset = SilverboxDataset(normalize=True)
        window_starts = range(0, num_steps - seq_len + 1, stride_window)
        u_train_seq = torch.stack([
            dataset.u_train[start:start + seq_len] for start in window_starts
        ]).view(-1, seq_len, dataset.u_train.shape[-1])
        y_train_seq = torch.stack([
            dataset.y_train[start:start + seq_len] for start in window_starts
        ]).view(-1, seq_len, dataset.y_train.shape[-1])


    case "vdp":
        all_u_windows, all_y_windows = [], []

        datasetFree = VDPDataset(csv_path='data/vanderpolDataFree.csv', normalize=False)
        datasetSine = VDPDataset(csv_path='data/vanderpolDataSine.csv', normalize=False)
        datasetMultisine = VDPDataset(csv_path='data/vanderpolDataMultisine.csv', normalize=False)
        datasetSweep = VDPDataset(csv_path='data/vanderpolDataSweep.csv', normalize=False)

        datasets = [datasetFree, datasetSine, datasetMultisine, datasetSweep]
        dataset = datasets[0]  # reference for dt/warmup_window/normalization stats, shared across all 4 files

        for ds in datasets:
            num_steps = ds.u_train.size(0)
            starts = range(0, num_steps - seq_len + 1, stride_window)
            all_u_windows.append(torch.stack([ds.u_train[s:s+seq_len] for s in starts]))
            all_y_windows.append(torch.stack([ds.y_train[s:s+seq_len] for s in starts]))

        u_train_seq = torch.cat(all_u_windows, dim=0)
        y_train_seq = torch.cat(all_y_windows, dim=0)
    case _:
        raise ValueError(f"Unknown case_name: {case_name}")

train_dataset = torch.utils.data.TensorDataset(u_train_seq, y_train_seq)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_steps = dataset.u_train.size(0)
if seq_len > num_steps:
    raise ValueError(f"Sequence length {seq_len} is larger than the dataset length {num_steps}.")
if stride_window <= 0:
    raise ValueError("stride_window must be greater than zero.")


subnetwork_shape = [32]
state_kan = FullStateNonlinearityRSKAN(
    2 + 1,  # input size (state_dim + control input)
    [2],
    2,  # output size (state correction dimension)
    subnetwork_shape = subnetwork_shape,
    masks = [
        [ [1,1],
          [1,1],
          [0,0] ],
          [[0, 1],
           [0, 1],
           ]
    ],
    residual_connection=False,
    zero_final_layer=False,
)
    # masks = [
    #     [ [1,1,0],
    #       [1,0,1],
    #       [0,0,0] ],
    #       [[0, 1],
    #        [0, 1],
    #        [0, 1]
    #        ]
    # ],
    #     masks = [
    #     [ [1],
    #       [1],
    #       [0] ],
    #       [[0, 1]
    #        ]
    # ],
model = Interpretable2DModel(device=device, case_name=case_name, dt=dataset.dt, linear_trainable=True, state_kan=state_kan).to(device)

# state_dict = torch.load(f"./interpretable_models/{case_name}.pth", map_location=device)
# state_dict = torch.load(f"./interpretable_models/vdp_0.0832_0.0444_0.0656_0.0593_[32]good2mask2noresidual.pth", map_location=device)
# model.load_state_dict(state_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-3, end_factor=1.0, total_iters=lr_warmup_epochs
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs - lr_warmup_epochs
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[lr_warmup_epochs],
)
loss = torch.nn.MSELoss()
# loss = torch.nn.HuberLoss(reduction='mean', delta=0.1)

loss_curve = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (u_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
        optimizer.zero_grad()

        model_output = []
        diverged = torch.zeros(u_batch.size(0), dtype=torch.bool, device=device)
        valid_mask = []
        # state = torch.zeros((u_batch.size(0), 2), device=device, dtype=torch.float32)
        # state = torch.tensor([[2.0, 0.0]] * u_batch.size(0), device=device, dtype=torch.float32)
        # x1_0 = y_batch[:, 0, 0]
        # x2_0 = (y_batch[:, 1, 0] - y_batch[:, 0, 0]) / dataset.dt

        # central difference approximation for x2_0
        x1_0 = y_batch[:, 1, 0]
        x2_0 = (y_batch[:, 2, 0] - y_batch[:, 0, 0]) / (2 * dataset.dt)


        state = torch.stack([x1_0, x2_0], dim=1).to(device)


        warmup_window = dataset.warmup_window if enable_warmup else 0
        if warmup_window >= seq_len:
            raise ValueError("warmup_window must be smaller than seq_len")

        for t in range(1, u_batch.size(1)):
            u_t = u_batch[:, t, :]
            next_state, y_pred = model(state, u_t)
            next_state = torch.clamp(next_state, min=-1000.0, max=1000.0)
            # next_state = 1000.0 * torch.tanh(next_state / 1000.0)
            if not torch.isfinite(next_state).all() or not torch.isfinite(y_pred).all():
                print(f"Non-finite rollout at epoch {epoch+1}, batch {batch_idx+1}, step {t+1}. Skipping this batch.")
                model_output = None
                break



            newly_diverged = (next_state.abs() > 10.0).any(dim=1)
            diverged = diverged | newly_diverged
            valid_mask.append(~diverged)

            model_output.append(y_pred)
            state = torch.where(diverged.unsqueeze(-1), state.detach(), next_state)

        if model_output is None:
            continue

        model_output = torch.stack(model_output, dim=1)
        valid_mask = torch.stack(valid_mask, dim=1)
        # loss_value = loss(model_output[:, warmup_window:], y_batch[:, 1+warmup_window:])

        sequence_error = (model_output[:, warmup_window:] - y_batch[:, 1+warmup_window:]) ** 2
        n_valid = valid_mask[:, warmup_window:].sum(dim=1)
        seq_ok = n_valid > 0

        per_sequence_loss = (sequence_error * valid_mask[:, warmup_window:].unsqueeze(-1)).sum(dim=1) / n_valid.clamp(min=1)
        loss_value = per_sequence_loss[seq_ok].mean() if seq_ok.any() else torch.tensor(0.0, device=device)

        if not torch.isfinite(loss_value):
            print(f"Non-finite loss at epoch {epoch+1}, batch {batch_idx+1}. Skipping this batch.")
            continue

        loss_value.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if not torch.isfinite(gradient_norm):
            print(f"Non-finite gradients at epoch {epoch+1}, batch {batch_idx+1}. Skipping this batch.")
            optimizer.zero_grad()
            continue

        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            print(f"Non-finite parameters after epoch {epoch+1}, batch {batch_idx+1}. Stopping training.")
            raise FloatingPointError("Optimizer produced non-finite model parameters.")
        epoch_loss += loss_value.item()
        if report_loss_curve:
            loss_curve.append(loss_value.item())
    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")

if report_loss_curve:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(loss_curve)
    plt.title("Training Loss Curve")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(f"./figures/{case_name}_loss_curve.png")

test_results = free_run_x1_nrmse(model, dataset)
print(test_results)

name = f"{test_results['free']:.4f}_{test_results['sine']:.4f}_{test_results['multisine']:.4f}_{test_results['sweep']:.4f}"

torch.save(model.state_dict(), f"./interpretable_models/{case_name}_{name}_{subnetwork_shape}.pth")
print(f"eigenvalues of A: {torch.linalg.eigvals(model.A)}")
print(model.A)
print(model.B)
print(model.C)
print(model.D)
# print(model.state_dict())
u_plot = torch.stack([d.u_plot for d in datasets], dim=1)  # [T, num_datasets, input_dim]
starting_state_plot = torch.cat([d.starting_state_plot for d in datasets], dim=0)  # [num_datasets, state_dim]
model.plot(u_plot, starting_state=starting_state_plot, warmup_window=dataset.warmup_window, attribution_score_alpha = True, sample=False, tick=True, folder="./figures")
