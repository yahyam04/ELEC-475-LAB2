"""
ELEC 475: Lab 2 - Automated Ablation Ladder Runner
Runs the 7-model ladder from Table 7.3 (and optional extra seeds) by calling train.py, writing one
results/<key>__<dataset>__s<seed>.json per run.

Examples:
    python run_experiments.py --dataset cifar10 --seeds 42                # single-seed ladder
    python run_experiments.py --dataset cifar10 --seeds 42 123 999        # multi-seed (Mean +/- Std)
    python run_experiments.py --dataset tiny_imagenet --seeds 42 --epochs 10
    python run_experiments.py --dry-run                                   # print commands only

Rows 1-5 and 7 are trained WITHOUT augmentation (--no-augment) so that Table 7.3a's
"Data Augmentation: None" column is true; only row 6 uses the augmentation pipeline.
"""
import argparse
import subprocess
import sys
import os

# key -> (train.py model, norm, use augmentation)
LADDER = {
    "lenet":          ("lenet",        None,   False),
    "lenet_modern":   ("lenet_modern", None,   False),
    "alexnet_none":   ("alexnet",      "none", False),
    "alexnet_lrn":    ("alexnet",      "lrn",  False),
    "alexnet_bn":     ("alexnet",      "bn",   False),
    "alexnet_bn_aug": ("alexnet",      "bn",   True),
    "zoo_alexnet":    ("zoo_alexnet",  None,   False),
}


def build_command(key, dataset, seed, epochs, extra):
    model, norm, aug = LADDER[key]
    run_name = f"{key}__{dataset}__s{seed}"
    cmd = [sys.executable, "train.py", "--model", model, "--dataset", dataset,
           "--epochs", str(epochs), "--seed", str(seed), "--run-name", run_name,
           "--save-model", f"checkpoints/{run_name}.pth",
           "--plot-file", f"plots/loss_{run_name}.png"]
    if norm is not None:
        cmd += ["--norm", norm]
    if not aug:
        cmd.append("--no-augment")
    return cmd + extra, run_name


def main():
    parser = argparse.ArgumentParser(description="ELEC 475 Lab 2 ablation ladder runner")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "tiny_imagenet"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=None, help="Default: 15 for CIFAR-10, 10 for Tiny ImageNet")
    parser.add_argument("--models", nargs="+", default=list(LADDER), choices=list(LADDER),
                        help="Subset of the ladder to run")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose results JSON already exists")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="Anything after --extra is forwarded to train.py (e.g. --extra --early-stopping)")
    args = parser.parse_args()

    epochs = args.epochs or (15 if args.dataset == "cifar10" else 10)
    for seed in args.seeds:
        for key in args.models:
            cmd, run_name = build_command(key, args.dataset, seed, epochs, args.extra)
            if args.skip_existing and os.path.exists(os.path.join("results", f"{run_name}.json")):
                print(f"[skip] {run_name}")
                continue
            print("\n$ " + " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
