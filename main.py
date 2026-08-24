import torch
from interpretable2DModel import Interpretable2DModel
from silverboxDataset import SilverboxDataset
from tqdm import tqdm


device = 'cpu'



silverbox_dataset = SilverboxDataset(normalize=True)
model = Interpretable2DModel(device=device, dt=silverbox_dataset.dt).to(device)

epochs = 10
seq_len = 256
learning_rate = 1e-2

num_steps = silverbox_dataset.u_train.size(0)
batch_size = num_steps // seq_len // 5 if num_steps // seq_len // 5 > 0 else 1

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
loss = torch.nn.MSELoss()

num_sequences = num_steps // seq_len
if num_sequences == 0:
    raise ValueError(f"Sequence length {seq_len} is larger than the dataset length {num_steps}.")

trimmed_steps = num_sequences * seq_len
u_train_seq = silverbox_dataset.u_train[:trimmed_steps].view(num_sequences, seq_len, -1)
y_train_seq = silverbox_dataset.y_train[:trimmed_steps].view(num_sequences, seq_len, -1)

train_dataset = torch.utils.data.TensorDataset(u_train_seq, y_train_seq)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (u_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
        optimizer.zero_grad()

        model_output = []
        state = torch.zeros((u_batch.size(0), 2), device=device, dtype=torch.double)

        for t in range(u_batch.size(1)):
            u_t = u_batch[:, t, :]
            next_state, y_pred = model(state, u_t)
            model_output.append(y_pred)
            state = next_state

        model_output = torch.stack(model_output, dim=1)
        loss_value = loss(model_output, y_batch)
        if not loss_value.requires_grad:
            raise RuntimeError(
                "Loss has no grad_fn. Check that sequence length is > 1 and outputs depend on trainable parameters across time."
            )
        loss_value.backward()
        optimizer.step()
        epoch_loss += loss_value.item()

    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")

    if (epoch + 1) % 99999 == 0:
        model.eval()
        with torch.no_grad():
            test_state = torch.zeros((1, 2), device=device, dtype=torch.double)
            test_output = []
            for t in range(silverbox_dataset.u_test.size(0)):
                u_t = silverbox_dataset.u_test[t].unsqueeze(0)
                next_state, y_pred = model(test_state, u_t)
                test_output.append(y_pred)
                test_state = next_state

            test_output = torch.cat(test_output, dim=0)
            test_loss_value = loss(test_output, silverbox_dataset.y_test)
            print(f"Test Loss after epoch {epoch+1}: {test_loss_value.item()}")

#print denormalized test and train RMSE
model.eval()
with torch.no_grad():
    test_state = torch.zeros((1, 2), device=device, dtype=torch.double)
    test_output = []
    for t in range(silverbox_dataset.u_test.size(0)):
        u_t = silverbox_dataset.u_test[t].unsqueeze(0)
        next_state, y_pred = model(test_state, u_t)
        test_output.append(y_pred)
        test_state = next_state

    test_output = torch.cat(test_output, dim=0)

    warmup_window = silverbox_dataset.warmup_window
    # Denormalize the predictions and true values
    y_pred_denorm = test_output[warmup_window:] * silverbox_dataset.y_std + silverbox_dataset.y_mean
    y_true_denorm = silverbox_dataset.y_test[warmup_window:] * silverbox_dataset.y_std + silverbox_dataset.y_mean

    rmse = torch.sqrt(torch.mean((y_pred_denorm - y_true_denorm) ** 2))
    print(f"Denormalized test RMSE: {rmse.item()}")

    # train_state = torch.zeros((silverbox_dataset.u_train.size(0), 2), device=device, dtype=torch.double)
    # train_output = []
    # for t in range(silverbox_dataset.u_train.size(1)):
    #     u_t = silverbox_dataset.u_train[:, t].unsqueeze(1)
    #     next_state, y_pred = model(train_state, u_t)
    #     train_output.append(y_pred)
    #     train_state = next_state

    # train_output = torch.stack(train_output, dim=1)

    # # Denormalize the predictions and true values
    # y_pred_train_denorm = train_output * silverbox_dataset.y_std + silverbox_dataset.y_mean
    # y_true_train_denorm = silverbox_dataset.y_train * silverbox_dataset.y_std + silverbox_dataset.y_mean

    # rmse_train = torch.sqrt(torch.mean((y_pred_train_denorm - y_true_train_denorm) ** 2))
    # print(f"Denormalized train RMSE: {rmse_train.item()}")


print(f"eigenvalues of A: {torch.linalg.eigvals(model.A)}")
print(model.A)
print(model.B)
print(model.C)
print(model.D)
