"""
ELEC 475: Lab 2 - Plotting & Visualization Utilities
"""
import os
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "./plots/training_curves.png",
    model_name: str = "Model"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--s', label='Val Loss')
    plt.title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'g--^', label='Val Accuracy')
    plt.title(f"{model_name.upper()} - Top-1 Accuracy (%)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves successfully saved to: {save_path}")

def plot_ablation_summary(
    ablation_results: Dict[str, Dict[str, List[float]]],
    save_path: str = "./plots/ablation_summary.png"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    styles = ['-', '--', '-.', ':', '-', '--', '-.']

    for idx, (model_label, data) in enumerate(ablation_results.items()):
        val_accs = data.get("val_accs", [])
        epochs = range(1, len(val_accs) + 1)
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        plt.plot(epochs, val_accs, linestyle=style, marker='o', color=color, label=model_label, linewidth=2)

    plt.title("ELEC 475: Master Ablation Study (Validation Accuracy)", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Master ablation summary plot saved to: {save_path}")


# ----------------------------------------------------------------------------------------------
# Ablation plot from saved results JSONs (written by train.py)
#   python -m utils.visualization --plot-ablation [--dataset cifar10] [--results-dir results]
# ----------------------------------------------------------------------------------------------
LADDER_LABELS = {
    "lenet":          "1. Base LeNet-5",
    "lenet_modern":   "2. Modernized LeNet",
    "alexnet_none":   "3. AlexNet (No Norm)",
    "alexnet_lrn":    "4. AlexNet + LRN",
    "alexnet_bn":     "5. AlexNet + BatchNorm",
    "alexnet_bn_aug": "6. AlexNet + BN + Aug",
    "zoo_alexnet":    "7. PyTorch Zoo AlexNet",
}


def load_ablation_histories(results_dir: str = "./results", dataset: str = "cifar10") -> Dict[str, Dict[str, List[float]]]:
    """
    Loads results/<key>__<dataset>__s<seed>.json files and returns, per ladder entry,
    the validation-accuracy curve averaged over every available seed (truncated to the shortest run).
    """
    import glob
    import json
    import numpy as np

    grouped: Dict[str, List[List[float]]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, f"*__{dataset}__s*.json"))):
        key = os.path.basename(path).split("__")[0]
        if key not in LADDER_LABELS:
            continue
        with open(path) as f:
            grouped.setdefault(key, []).append(json.load(f)["history"]["val_accs"])

    out: Dict[str, Dict[str, List[float]]] = {}
    for key in LADDER_LABELS:  # keep ladder order
        if key in grouped:
            n = min(len(c) for c in grouped[key])
            mean_curve = np.mean([c[:n] for c in grouped[key]], axis=0).tolist()
            label = f"{LADDER_LABELS[key]} (n={len(grouped[key])})"
            out[label] = {"val_accs": mean_curve}
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ELEC 475: Lab 2 plotting utilities")
    parser.add_argument("--plot-ablation", action="store_true", help="Plot the master ablation validation-accuracy curves")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "tiny_imagenet"])
    parser.add_argument("--save-path", type=str, default="./plots/ablation_summary.png")
    args = parser.parse_args()

    if args.plot_ablation:
        histories = load_ablation_histories(args.results_dir, args.dataset)
        if not histories:
            raise SystemExit(f"No results found in '{args.results_dir}' for dataset '{args.dataset}'. "
                             f"Run run_experiments.py (or train.py) first.")
        plot_ablation_summary(histories, save_path=args.save_path)
    else:
        parser.print_help()
