import torch
from interpretable2DModel import Interpretable2DModel
from silverboxDataset import SilverboxDataset
from tqdm import tqdm
from rskanSS import FullStateNonlinearityRSKAN
from vaderpolDataset import VDPDataset

# torch.autograd.set_detect_anomaly(True)
device = 'cpu'

case_name = "vdp" # silverbox or vdp

match case_name:
    case "silverbox":
        dataset = SilverboxDataset(normalize=True)
    case "vdp":
        dataset = VDPDataset(normalize=True)
    case _:
        raise ValueError(f"Unknown case_name: {case_name}")

subnetwork_shape = [20,20]
state_kan = FullStateNonlinearityRSKAN(
    2 + 1,  # input size (state_dim + control input)
    [3],
    2,  # output size (state correction dimension)
    subnetwork_shape = subnetwork_shape,
    masks = [
        [ [1, 1, 0],
          [1, 0, 1],
          [0, 0, 0] ],
          [[0, 1],
           [0, 1],
           [0, 1]]
    ],
    residual_connection=False,
    zero_final_layer=False,
)

model = Interpretable2DModel(device=device, case_name=case_name, dt=dataset.dt, linear_trainable=True, state_kan=state_kan).to(device)

# state_dict = torch.load(f"./interpretable_models/{case_name}.pth", map_location=device)
state_dict = torch.load(f"./interpretable_models/vdp_0.791875_500.pth", map_location=device)
model.load_state_dict(state_dict)

epochs = 150
seq_len = 1000
learning_rate = 1e-3
enable_warmup = True

num_steps = dataset.u_train.size(0)
batch_size = num_steps // seq_len // 15 if num_steps // seq_len // 15 > 0 else 1

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
loss = torch.nn.MSELoss()

num_sequences = num_steps // seq_len
if num_sequences == 0:
    raise ValueError(f"Sequence length {seq_len} is larger than the dataset length {num_steps}.")

trimmed_steps = num_sequences * seq_len
u_train_seq = dataset.u_train[:trimmed_steps].view(num_sequences, seq_len, -1)
y_train_seq = dataset.y_train[:trimmed_steps].view(num_sequences, seq_len, -1)

train_dataset = torch.utils.data.TensorDataset(u_train_seq, y_train_seq)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (u_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
        optimizer.zero_grad()

        model_output = []
        state = torch.zeros((u_batch.size(0), 2), device=device, dtype=torch.float32)

        warmup_window = dataset.warmup_window if enable_warmup else 0
        if warmup_window >= seq_len:
            raise ValueError("warmup_window must be smaller than seq_len")

        for t in range(u_batch.size(1)):
            u_t = u_batch[:, t, :]
            next_state, y_pred = model(state, u_t)
            model_output.append(y_pred)
            state = next_state

        model_output = torch.stack(model_output, dim=1)
        loss_value = loss(model_output[:, warmup_window:], y_batch[:, warmup_window:])
        if torch.isinf(loss_value).any():
            print(f"Inf loss encountered at epoch {epoch+1}, batch {batch_idx+1}. Skipping this batch.")
            continue
        if torch.isnan(loss_value).any():
            print(f"NaN loss encountered at epoch {epoch+1}, batch {batch_idx+1}. Skipping this batch.")
            continue


        loss_value.backward()
        optimizer.step()
        epoch_loss += loss_value.item()

    # scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")
    if (epoch + 1) % 500 == 0:
        torch.save(model.state_dict(), f"./interpretable_models/{case_name}_{epoch_loss/len(train_loader):.6f}.pth")
        print(f"Model saved after epoch {epoch+1}")

    if (epoch + 1) % 99999 == 0:
        model.eval()
        with torch.no_grad():
            test_state = torch.zeros((1, 2), device=device, dtype=torch.float32)
            test_output = []
            for t in range(dataset.u_test.size(0)):
                u_t = dataset.u_test[t].unsqueeze(0)
                next_state, y_pred = model(test_state, u_t)
                test_output.append(y_pred)
                test_state = next_state

            test_output = torch.cat(test_output, dim=0)
            test_loss_value = loss(test_output, dataset.y_test)
            print(f"Test Loss after epoch {epoch+1}: {test_loss_value.item()}")

#print denormalized test and train RMSE
model.eval()
with torch.no_grad():
    test_state = torch.zeros((1, 2), device=device, dtype=torch.float32)
    test_output = []
    for t in range(dataset.u_test.size(0)):
        u_t = dataset.u_test[t].unsqueeze(0)
        next_state, y_pred = model(test_state, u_t)
        test_output.append(y_pred)
        test_state = next_state

    test_output = torch.cat(test_output, dim=0)

    warmup_window = dataset.warmup_window if enable_warmup else 0
    # Denormalize the predictions and true values
    y_pred_denorm = test_output[warmup_window:] * dataset.y_std + dataset.y_mean
    y_true_denorm = dataset.y_test[warmup_window:] * dataset.y_std + dataset.y_mean

    rmse = torch.sqrt(torch.mean((y_pred_denorm - y_true_denorm) ** 2))
    print(f"Denormalized test RMSE: {rmse.item()}")

    # train_state = torch.zeros((dataset.u_train.size(0), 2), device=device, dtype=torch.float32)
    # train_output = []
    # for t in range(dataset.u_train.size(1)):
    #     u_t = dataset.u_train[:, t].unsqueeze(1)
    #     next_state, y_pred = model(train_state, u_t)
    #     train_output.append(y_pred)
    #     train_state = next_state

    # train_output = torch.stack(train_output, dim=1)

    # # Denormalize the predictions and true values
    # y_pred_train_denorm = train_output * dataset.y_std + dataset.y_mean
    # y_true_train_denorm = dataset.y_train * dataset.y_std + dataset.y_mean

    # rmse_train = torch.sqrt(torch.mean((y_pred_train_denorm - y_true_train_denorm) ** 2))
    # print(f"Denormalized train RMSE: {rmse_train.item()}")

torch.save(model.state_dict(), f"./interpretable_models/{case_name}_{epoch_loss/len(train_loader):.6f}.pth")
print(f"eigenvalues of A: {torch.linalg.eigvals(model.A)}")
print(model.A)
print(model.B)
print(model.C)
print(model.D)
model.plot(dataset.u_test, warmup_window=dataset.warmup_window, attribution_score_alpha = False, sample=True, tick=True, folder="./figures")
