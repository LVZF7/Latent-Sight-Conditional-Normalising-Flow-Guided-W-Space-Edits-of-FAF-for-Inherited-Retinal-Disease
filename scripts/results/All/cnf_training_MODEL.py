import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt

# Load JSON logs from a file
def load_json_logs(log_path: Path) -> Dict[str, Any]:
    with open(log_path, "r") as f:
        return json.load(f)

# Helper to get value from multiple possible keys
def get_val(entry: Dict[str, Any], keys: List[str], default=np.nan):
    for k in keys:
        if k in entry:
            return entry[k]
    return default

# Read training history and extract metrics
def read_history(history):
    epochs, nlls, bpds, lrs = [], [], [], []
    epoch_counter = 0

    for entry in history or []:
        epoch = int(entry.get("epoch", epoch_counter + 1))
        nll  = float(get_val(entry, ["NLL", "negative_log_likelihood"], np.nan))
        bpd  = float(get_val(entry, ["bits_per_dim", "bpd"], np.nan))
        lr   = float(get_val(entry, ["lr", "learning_rate"], np.nan))

        epochs.append(epoch)
        nlls.append(nll)
        bpds.append(bpd)
        lrs.append(lr)
        epoch_counter = epoch

    epochs = np.asarray(epochs, dtype=int)
    nlls   = np.asarray(nlls, dtype=float)
    bpds   = np.asarray(bpds, dtype=float)
    lrs    = np.asarray(lrs, dtype=float)

    if len(epochs) == 0 or (np.all(np.isnan(nlls)) and np.all(np.isnan(bpds))):
        raise ValueError("No usable history entries (need NLL or bits_per_dim).")

    return epochs, nlls, bpds, lrs

# Find the best epoch based on NLL or bits per dim
def find_best(epochs, nlls, bpds):
    if np.all(np.isnan(nlls)) and np.any(~np.isnan(bpds)):
        idx = int(np.nanargmin(bpds))
    else:
        idx = int(np.nanargmin(nlls))
    return {
        "idx": idx,
        "epoch": int(epochs[idx]),
        "best_nll": float(nlls[idx]),
        "best_bpd": float(bpds[idx]),
    }

# Plot training metrics
def plot_metrics(epochs, nlls, bpds, lrs, best, out_png, title = "CNF Training Metrics"):
    plt.figure(figsize=(15, 5))

    # NLL
    ax = plt.subplot(1, 3, 1)
    ax.plot(epochs, nlls, marker='o')
    if not np.isnan(best["best_nll"]):
        ax.scatter([best["epoch"]], [best["best_nll"]], s=30)
        ax.annotate(f"best at {best['epoch']}", (best["epoch"], best["best_nll"]), xytext=(5,5), textcoords='offset points')
    ax.set_title('Negative Log Likelihood')
    ax.set_xlabel('Epoch'); ax.set_ylabel('NLL')

    # Bits per dim
    ax = plt.subplot(1, 3, 2)
    ax.plot(epochs, bpds, marker='o')
    if not np.isnan(best["best_bpd"]):
        ax.scatter([best["epoch"]], [best["best_bpd"]], s=30)
        ax.annotate(f"best at {best['epoch']}", (best["epoch"], best["best_bpd"]), xytext=(5,5), textcoords='offset points')
    ax.set_title('Bits per Dimension')
    ax.set_xlabel('Epoch'); ax.set_ylabel('bits/dim')

    # Learning rate
    ax = plt.subplot(1, 3, 3)
    ax.plot(epochs, lrs, marker='o')
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR')

    plt.suptitle(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"Saved plot: {out_png}")

# Write summary CSV with best metrics
def write_summary_csv(csv_path, log_path, best):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["run", "best_epoch", "best_nll", "best_bits_per_dim"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([str(log_path), best["epoch"], f"{best['best_nll']:.6f}", f"{best['best_bpd']:.6f}"])
    print(f"Appended summary row to: {csv_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_path", type=Path)
    p.add_argument("--output_plot_path", type=Path, default=Path("training_metrics.png"))
    p.add_argument("--output_csv_path", type=Path, default=Path("training_summary.csv"))
    p.add_argument("--title", default="CNF Training (StyleFlow)")
    args = p.parse_args()

    logs = load_json_logs(args.log_path)
    epochs, nlls, bpds, lrs = read_history(logs.get("history", []))
    best = find_best(epochs, nlls, bpds)

    plot_metrics(epochs, nlls, bpds, lrs, best, args.output_plot_path, args.title)
    write_summary_csv(args.output_csv_path, args.log_path, best)


if __name__ == "__main__":
    main()


"""
Usage:
python scripts/results/All/cnf_training_MODEL.py \
    --log_path experiment/gene_36class/flow/checkpoints/training_summary.json \
    --output_plot_path experiment/gene_36class/flow/checkpoints/training_metrics.png \
    --output_csv_path experiment/gene_36class/flow/checkpoints/training_summary.csv \
    --title "StyleFlow CNF Training (gene_36class)"
    
python scripts/results/All/cnf_training_MODEL.py \
    --log_path experiment/laterality/flow/checkpoints/training_summary.json \
    --output_plot_path experiment/laterality/flow/checkpoints/training_metrics.png \
    --output_csv_path experiment/laterality/flow/checkpoints/training_summary.csv \
    --title "StyleFlow CNF Training (laterality)"
"""