"""
Standalone evaluation bench for a trained VDP Interpretable2DModel checkpoint.

Loads a saved model and runs it against the last TEST_LEN rows of each vanderpol*.csv
excitation file (matching the train/test split used in main.py):
  - "free" is the file used for training -> its test slice is held-out.
  - "sine"/"multisine"/"sweep" were never trained on at all -> cross-excitation
    generalization check on the same-length tail slice.

For each trajectory it reports:
  - free-run (closed-loop) rollout error, on x1, x2 AND y
  - one-step-ahead (teacher-forced) error, to separate model-quality from error accumulation
  - FIT% / NRMSE / VAF alongside RMSE
  - time-domain and phase-portrait plots

It also runs two structure-specific interpretability checks:
  - a symbolic polynomial fit of every active KAN edge (degree + R^2)
  - a direct comparison of the learned state-correction surface against the
    hypothesized 2*x1^2*x2 identity ((x1^2+x2)^2 - x1^4 - x2^2), fitting the best
    scalar multiplier and reporting how much of the surface it explains.

Run: python test_bench.py --model ./interpretable_models/vdp_0.040905.pth
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from interpretable2DModel import Interpretable2DModel
from rskanSS import FullStateNonlinearityRSKAN
from vanderpolDataset import VDPDataset

DEVICE = "cpu"

DATA_FILES = {
    "free": "data/vanderpolDataFree.csv",
    "sine": "data/vanderpolDataSine.csv",
    "multisine": "data/vanderpolDataMultisine.csv",
    "sweep": "data/vanderpolDataSweep.csv",
}
TEST_LEN = 1500  # must match the split used in main.py
SUBNETWORK_SHAPE = [32]  # must match the shape used in main.py
FILE_NAME = "./interpretable_models/vdp_0.0516_0.0623_0.0720_0.0587_[20].pth"

# ----------------------------------------------------------------------------
# Model / data loading
# ----------------------------------------------------------------------------

def build_vdp_model(dataset, device=DEVICE):
    """Mirrors the architecture built in main.py. Keep in sync if that changes."""
    subnetwork_shape = SUBNETWORK_SHAPE
    state_kan = FullStateNonlinearityRSKAN(
        2 + 1,
        [2],
        2,
        subnetwork_shape=subnetwork_shape,
        masks=[
            [[1, 1],
             [1, 1],
             [0, 0]],
            [[0, 1],
             [0, 1]],
        ],
        residual_connection=False,
        zero_final_layer=True,
    )
    return Interpretable2DModel(
        device=device, case_name="vdp", dt=dataset.dt,
        linear_trainable=False, state_kan=state_kan,
    ).to(device)


def load_model(model_path, dataset, device=DEVICE):
    model = build_vdp_model(dataset, device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_raw_trajectory(csv_path, dataset, split="all", test_len=TEST_LEN):
    """Loads u, x1, x2 from a vanderpol*.csv and normalizes with the reference dataset's stats."""
    data = pd.read_csv(csv_path)
    u = torch.tensor(data["u"].values, dtype=torch.float32).view(-1, 1)
    x1 = torch.tensor(data["x1"].values, dtype=torch.float32).view(-1, 1)
    x2 = torch.tensor(data["x2"].values, dtype=torch.float32).view(-1, 1)
    dt = float(data["time"][1] - data["time"][0])

    if split == "test":
        u, x1, x2 = u[-test_len:], x1[-test_len:], x2[-test_len:]
    elif split == "train":
        u, x1, x2 = u[:-test_len], x1[:-test_len], x2[:-test_len]

    u_n = (u - dataset.u_mean) / dataset.u_std
    x1_n = (x1 - dataset.y_mean) / dataset.y_std
    x2_n = x2 / dataset.y_std  # x2 isn't independently normalized in VDPDataset; matches the normalize=False default

    return {"u": u_n, "x1": x1_n, "x2": x2_n, "dt": dt}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()


def nrmse(pred, true):
    return rmse(pred, true) / (true.std().item() + 1e-12)


def fit_percentage(pred, true):
    num = torch.norm(true - pred)
    den = torch.norm(true - true.mean()) + 1e-12
    return (100.0 * (1.0 - num / den)).item()


def vaf(pred, true):
    resid_var = torch.var(true - pred, unbiased=False)
    true_var = torch.var(true, unbiased=False) + 1e-12
    return (100.0 * (1.0 - resid_var / true_var)).item()


def metric_row(pred, true, label):
    return {
        "signal": label,
        "rmse": rmse(pred, true),
        "nrmse": nrmse(pred, true),
        "fit_pct": fit_percentage(pred, true),
        "vaf_pct": vaf(pred, true),
    }


# ----------------------------------------------------------------------------
# Rollouts
# ----------------------------------------------------------------------------

def free_run_rollout(model, u_seq, x0, device=DEVICE):
    """Closed-loop simulation: no ground-truth state is fed back in, ever. No clamping,
    so real divergence is visible instead of being masked (unlike the training loop)."""
    model.eval()
    state = x0.to(device)
    states = [state.squeeze(0).clone()]
    ys = []
    with torch.no_grad():
        for t in range(u_seq.size(0)):
            u_t = u_seq[t].unsqueeze(0).to(device)
            next_state, y_pred = model(state, u_t)
            ys.append(y_pred.squeeze(0).clone())
            states.append(next_state.squeeze(0).clone())
            state = next_state
    states = torch.stack(states)  # [T+1, 2]
    ys = torch.stack(ys)  # [T, 1]
    return states, ys


def one_step_predictions(model, u_seq, x1_true, x2_true, device=DEVICE):
    """Teacher-forced: true state fed in at every step, isolates per-step model error
    from rollout error accumulation."""
    model.eval()
    true_state = torch.cat([x1_true, x2_true], dim=-1).to(device)  # [T, 2]
    T = u_seq.size(0)
    preds_next, ys = [], []
    with torch.no_grad():
        for t in range(T - 1):
            s_t = true_state[t].unsqueeze(0)
            u_t = u_seq[t].unsqueeze(0).to(device)
            next_state, y_pred = model(s_t, u_t)
            preds_next.append(next_state.squeeze(0))
            ys.append(y_pred.squeeze(0))
    return torch.stack(preds_next), torch.stack(ys)


def free_run_x1_nrmse(model, reference_dataset, dataset_names=("free", "sine", "multisine", "sweep"),
                       device=DEVICE):
    """Free-run (closed-loop) NRMSE on x1(=y) for each named dataset's held-out test slice.

    Lightweight alternative to the full bench (no plots/files) meant to be called from
    main.py, e.g. after training, to get a quick per-checkpoint generalization summary.

    Args:
        model: a loaded/trained Interpretable2DModel.
        reference_dataset: the VDPDataset used to build/train the model (supplies
            u_mean/u_std/y_mean/y_std for normalization).
        dataset_names: subset of DATA_FILES keys to evaluate.
        device: torch device.

    Returns:
        dict[str, float]: {dataset_name: free_run_x1_nrmse}
    """
    was_training = model.training
    model.eval()
    results = {}
    for name in dataset_names:
        traj = load_raw_trajectory(DATA_FILES[name], reference_dataset, split="test")
        x0 = torch.tensor([[traj["x1"][0].item(), traj["x2"][0].item()]], dtype=torch.float32)
        fr_states, _ = free_run_rollout(model, traj["u"], x0, device)
        fr_x1 = fr_states[1:, 0:1]
        results[name] = nrmse(fr_x1, traj["x1"])
    if was_training:
        model.train()
    return results


# ----------------------------------------------------------------------------
# Per-trajectory evaluation + plots
# ----------------------------------------------------------------------------

def evaluate_trajectory(model, name, traj, out_dir, device=DEVICE):
    u_seq, x1_true, x2_true = traj["u"], traj["x1"], traj["x2"]
    dt = traj["dt"]
    T = u_seq.size(0)
    t_axis = np.arange(T) * dt

    x0 = torch.tensor([[x1_true[0].item(), x2_true[0].item()]], dtype=torch.float32)

    fr_states, fr_y = free_run_rollout(model, u_seq, x0, device)
    fr_x1, fr_x2 = fr_states[1:, 0:1], fr_states[1:, 1:2]

    os_next, os_y = one_step_predictions(model, u_seq, x1_true, x2_true, device)
    os_x1, os_x2 = os_next[:, 0:1], os_next[:, 1:2]

    rows = []
    rows.append({**metric_row(fr_x1, x1_true, "free_run_x1(=y)"), "mode": "free_run"})
    rows.append({**metric_row(fr_x2, x2_true, "free_run_x2"), "mode": "free_run"})
    rows.append({**metric_row(os_x1, x1_true[1:], "one_step_x1(=y)"), "mode": "one_step"})
    rows.append({**metric_row(os_x2, x2_true[1:], "one_step_x2"), "mode": "one_step"})
    for r in rows:
        r["trajectory"] = name

    traj_dir = os.path.join(out_dir, name)
    os.makedirs(traj_dir, exist_ok=True)

    # Time-domain: input, x1, x2 (free-run vs true)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t_axis, u_seq.numpy(), color="black", lw=0.8)
    axes[0].set_ylabel("u")
    axes[1].plot(t_axis, x1_true.numpy(), label="true", color="black", lw=1.0)
    axes[1].plot(t_axis, fr_x1.numpy(), label="free-run pred", color="tab:red", lw=1.0, alpha=0.8)
    axes[1].set_ylabel("x1 (=y)")
    axes[1].legend(fontsize=8)
    axes[2].plot(t_axis, x2_true.numpy(), label="true", color="black", lw=1.0)
    axes[2].plot(t_axis, fr_x2.numpy(), label="free-run pred", color="tab:red", lw=1.0, alpha=0.8)
    axes[2].set_ylabel("x2")
    axes[2].set_xlabel("time")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{name}: free-run rollout")
    fig.tight_layout()
    fig.savefig(os.path.join(traj_dir, "time_domain.png"), dpi=160)
    plt.close(fig)

    # Phase portrait
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x1_true.numpy(), x2_true.numpy(), color="black", lw=0.8, label="true")
    ax.plot(fr_x1.numpy(), fr_x2.numpy(), color="tab:red", lw=0.8, alpha=0.8, label="free-run pred")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"{name}: phase portrait")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(traj_dir, "phase_portrait.png"), dpi=160)
    plt.close(fig)

    # Rollout error growth (diagnoses divergence that training-time clamping would hide)
    err = torch.sqrt((fr_x1 - x1_true) ** 2 + (fr_x2 - x2_true) ** 2).squeeze(-1).numpy()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_axis, err, color="tab:red", lw=1.0)
    ax.set_xlabel("time")
    ax.set_ylabel("||state error||")
    ax.set_title(f"{name}: free-run state error growth")
    fig.tight_layout()
    fig.savefig(os.path.join(traj_dir, "error_growth.png"), dpi=160)
    plt.close(fig)

    max_err = float(err.max())
    if not np.isfinite(err).all() or max_err > 1e6:
        print(f"  [!] {name}: free-run rollout diverges (max ||error|| = {max_err:.3e})")

    return rows


# ----------------------------------------------------------------------------
# Interpretability diagnostics
# ----------------------------------------------------------------------------

def print_linear_system(model):
    print("\n--- Linear backbone ---")
    print("A:\n", model.A)
    print("B:\n", model.B)
    print("C:\n", model.C)
    print("D:\n", model.D)
    print("eigenvalues(A):", torch.linalg.eigvals(model.A))


def symbolic_edge_report(kan, max_degree=2, r2_threshold=0.995):
    """Fits each active KAN edge to the lowest-degree polynomial that reaches r2_threshold
    (or the best of max_degree if none does), using the activations saved by the last
    forward pass with save_activations=True."""
    rows = []
    for l, layer in enumerate(kan.layers):
        if layer.pre_activations is None or layer.post_activations is None:
            continue
        for i in range(layer.input_size):
            for j in range(layer.output_size):
                if layer.mask[i, j].item() == 0:
                    continue
                x = layer.pre_activations[:, i].detach().cpu().numpy()
                y = layer.post_activations[i, j, :].detach().cpu().numpy()
                if np.std(y) < 1e-6:
                    rows.append({"layer": l, "in": i, "out": j, "degree": None, "r2": None, "note": "inactive"})
                    continue
                best = None
                for deg in range(1, max_degree + 1):
                    coeffs = np.polyfit(x, y, deg)
                    y_fit = np.polyval(coeffs, x)
                    ss_res = np.sum((y - y_fit) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
                    r2 = 1.0 - ss_res / ss_tot
                    if best is None or r2 > best["r2"]:
                        best = {"degree": deg, "r2": r2, "coeffs": coeffs}
                    if r2 >= r2_threshold:
                        break
                rows.append({
                    "layer": l, "in": i, "out": j,
                    "degree": best["degree"], "r2": best["r2"],
                    "coeffs": np.round(best["coeffs"], 4).tolist(),
                    "note": "",
                })

    print("\n--- Symbolic edge fits (lowest degree reaching R^2 >= %.3f) ---" % r2_threshold)
    for r in rows:
        if r["degree"] is None:
            print(f"  layer {r['layer']} edge ({r['in']}->{r['out']}): inactive")
        else:
            print(f"  layer {r['layer']} edge ({r['in']}->{r['out']}): "
                  f"deg={r['degree']} R^2={r['r2']:.4f} coeffs(high->low)={r['coeffs']}")
    return rows


def multiplication_identity_check(model, out_dir, x1_range=(-3, 3), x2_range=(-3, 3), n=60, device=DEVICE):
    """Compares the learned x2-state-correction surface against the hypothesized
    2*x1^2*x2 identity ((x1^2+x2)^2 - x1^4 - x2^2), fitting the best scalar multiplier."""
    x1 = torch.linspace(*x1_range, n)
    x2 = torch.linspace(*x2_range, n)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
    state = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1).to(device)
    u = torch.zeros(state.size(0), 1, device=device)  # u isn't wired into this KAN (mask row is all-zero)

    model.eval()
    with torch.no_grad():
        correction = model.state_kan(state=state, u=u)
    corr_x2 = correction[:, 1].reshape(n, n).cpu().numpy()

    X1n, X2n = X1.numpy(), X2.numpy()
    target = X1n ** 2 * X2n
    c = float(np.sum(corr_x2 * target) / (np.sum(target * target) + 1e-12))
    resid = corr_x2 - c * target
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((corr_x2 - corr_x2.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    print("\n--- Multiplication identity check (learned x2-correction vs c * x1^2 * x2) ---")
    print(f"  best-fit c = {c:.5f}, R^2 = {r2:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, data_, title in zip(
        axes, [corr_x2, c * target, resid],
        ["learned correction", f"c * x1^2 * x2  (c={c:.4f})", "residual"],
    ):
        im = ax.pcolormesh(X1n, X2n, data_, shading="auto", cmap="RdBu_r",
                            vmin=-np.abs(corr_x2).max(), vmax=np.abs(corr_x2).max())
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Multiplication identity check: R^2={r2:.4f}")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "multiplication_identity_check.png"), dpi=160)
    plt.close(fig)

    return c, r2


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained VDP model checkpoint.")
    parser.add_argument("--model", type=str, default=FILE_NAME)
    parser.add_argument("--datasets", type=str, nargs="+", default=list(DATA_FILES.keys()),
                         choices=list(DATA_FILES.keys()))
    parser.add_argument("--out", type=str, default="./figures/test_bench")
    args = parser.parse_args()

    reference_dataset = VDPDataset(csv_path=DATA_FILES["free"], normalize=False)  # gives dt + normalization stats used at training time
    model = load_model(args.model, reference_dataset, DEVICE)

    print(f"Loaded model: {args.model}")
    print_linear_system(model)

    os.makedirs(args.out, exist_ok=True)
    all_rows = []
    all_states_for_symbolic = []

    for name in args.datasets:
        csv_path = DATA_FILES[name]
        traj = load_raw_trajectory(csv_path, reference_dataset, split="test")
        print(f"\nEvaluating '{name}' ({csv_path}, split=test, T={traj['u'].size(0)})")
        rows = evaluate_trajectory(model, name, traj, args.out, DEVICE)
        all_rows.extend(rows)
        for r in rows:
            print(f"  [{r['mode']:>9}] {r['signal']:<20} "
                  f"RMSE={r['rmse']:.5f}  NRMSE={r['nrmse']:.4f}  "
                  f"FIT%={r['fit_pct']:.2f}  VAF%={r['vaf_pct']:.2f}")
        all_states_for_symbolic.append(torch.cat([traj["x1"], traj["x2"]], dim=-1))

    summary = pd.DataFrame(all_rows)[["trajectory", "mode", "signal", "rmse", "nrmse", "fit_pct", "vaf_pct"]]
    summary_path = os.path.join(args.out, "summary_metrics.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved metrics summary to {summary_path}")

    # Symbolic edge fit + multiplication-identity check use activations from a broad
    # sweep across every evaluated excitation type, not just one trajectory.
    combined_states = torch.cat(all_states_for_symbolic, dim=0)
    combined_u = torch.zeros(combined_states.size(0), 1)
    with torch.no_grad():
        model.state_kan(state=combined_states, u=combined_u)
    symbolic_edge_report(model.state_kan.kan)

    multiplication_identity_check(model, args.out, device=DEVICE)


if __name__ == "__main__":
    main()
