"""
ELEC 475: Lab 2 - Report Table Generator
Reads results/*.json (written by train.py) and prints the report tables as Markdown:
  * Table 7.3a - architecture & computational complexity profile
  * Table 7.3b - predictive performance on CIFAR-10 and Tiny ImageNet (Mean +/- Std over seeds)
  * Generalization-gap table (clean train acc - val acc) for BN with vs. without augmentation
Saves everything to results/tables.md.

    python aggregate_results.py [--latency]

--latency additionally measures inference latency (ms, batch=1) with utils.profiler.benchmark_inference.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from utils.seed import calculate_mean_std, format_mean_std
from utils.visualization import LADDER_LABELS

META = {  # key: (conv depth, normalization & regularization, augmentation)
    "lenet":          (2, "None (Tanh + AvgPool)", "None"),
    "lenet_modern":   (2, "None (ReLU + Overlapping MaxPool)", "None"),
    "alexnet_none":   (5, "Dropout (p=0.5)", "None"),
    "alexnet_lrn":    (5, "LRN + Dropout (p=0.5)", "None"),
    "alexnet_bn":     (5, "BatchNorm2d + Dropout (p=0.5)", "None"),
    "alexnet_bn_aug": (5, "BatchNorm2d + Dropout (p=0.5)", "Pure PyTorch Aug"),
    "zoo_alexnet":    (5, "PyTorch Zoo (AdaptiveAvgPool2d)", "None"),
}


def load(results_dir):
    runs = defaultdict(list)  # (key, dataset) -> [result dicts]
    for path in sorted(glob.glob(os.path.join(results_dir, "*__*__s*.json"))):
        key, dataset = os.path.basename(path).split("__")[:2]
        with open(path) as f:
            runs[(key, dataset)].append(json.load(f))
    return runs


def ms(values, decimals=2):
    return format_mean_std(values, unit="", decimals=decimals) if len(values) > 1 else f"{values[0]:.{decimals}f}"


def measure_latency(key, num_classes=10, device=None):
    import torch
    from models import get_model
    from utils.profiler import benchmark_inference
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name, norm = {"lenet": ("lenet", "bn"), "lenet_modern": ("lenet_modern", "bn"),
                        "alexnet_none": ("alexnet", "none"), "alexnet_lrn": ("alexnet", "lrn"),
                        "alexnet_bn": ("alexnet", "bn"), "alexnet_bn_aug": ("alexnet", "bn"),
                        "zoo_alexnet": ("zoo_alexnet", "bn")}[key]
    model = get_model(model_name, num_classes=num_classes, norm_type=norm)
    latency, _ = benchmark_inference(model, input_size=(1, 3, 64, 64), device=device, reps=100, warmup=10)
    return latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--latency", action="store_true")
    args = parser.parse_args()

    runs = load(args.results_dir)
    if not runs:
        raise SystemExit("No results found. Run run_experiments.py first.")
    out = []

    # ---- Table 7.3a ----
    out.append("### Table 7.3a: Architectural Configuration & Computational Complexity (CIFAR-10 runs)\n")
    out.append("| Model Architecture | Conv Depth | Normalization & Regularization | Data Aug | Params (M) | FLOPs (M) | Mean Epoch Time (s) | Latency (ms) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for key, label in LADDER_LABELS.items():
        rs = runs.get((key, "cifar10"))
        if not rs:
            continue
        depth, reg, aug = META[key]
        t = ms([r["mean_epoch_time_s"] for r in rs])
        lat = f"{measure_latency(key):.2f}" if args.latency else "n/a (use --latency)"
        out.append(f"| {label} | {depth} | {reg} | {aug} | {rs[0]['params_m']:.3f} | {rs[0]['flops_m']:.2f} | {t} | {lat} |")

    # ---- Table 7.3b ----
    out.append("\n### Table 7.3b: Predictive Performance (final-epoch weights; Mean ± Std over seeds)\n")
    out.append("| Model Architecture | #Seeds (C10 / Tiny) | CIFAR-10 Val Acc (%) | CIFAR-10 Val Macro F1 (%) | Tiny ImageNet Val Acc (%) | Tiny ImageNet Val Macro F1 (%) |")
    out.append("|---|---|---|---|---|---|")
    for key, label in LADDER_LABELS.items():
        c = runs.get((key, "cifar10"), [])
        t = runs.get((key, "tiny_imagenet"), [])
        if not c and not t:
            continue
        def cell(rs, field):
            return ms([r["final"][field] for r in rs]) if rs else "—"
        out.append(f"| {label} | {len(c)} / {len(t)} | {cell(c,'val_acc')} | {cell(c,'val_macro_f1')} | {cell(t,'val_acc')} | {cell(t,'val_macro_f1')} |")

    # ---- Generalization gap ----
    out.append("\n### Generalization Gap Δ = Train Acc − Val Acc (eval mode, no augmentation)\n")
    out.append("| Dataset | Configuration | Train Acc (%) | Val Acc (%) | Δ (pts) | Val Macro F1 (%) |")
    out.append("|---|---|---|---|---|---|")
    for dataset in ("cifar10", "tiny_imagenet"):
        for key in ("alexnet_bn", "alexnet_bn_aug"):
            rs = runs.get((key, dataset))
            if not rs:
                continue
            f = lambda fld: ms([r["final"][fld] for r in rs])
            out.append(f"| {dataset} | {LADDER_LABELS[key]} | {f('train_acc')} | {f('val_acc')} | {f('gap')} | {f('val_macro_f1')} |")

    text = "\n".join(out)
    print(text)
    path = os.path.join(args.results_dir, "tables.md")
    with open(path, "w") as fh:
        fh.write(text + "\n")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
