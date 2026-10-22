### Extended Ablation: Learning Rate & Early Stopping (CIFAR-10, AlexNet + BN, no augmentation, seed 42)

| Experiment | Epochs run | Train Acc (%) | Val Acc (%) | Δ vs baseline (pts) | Val Macro F1 (%) | Gap Δ (pts) | Mean Epoch Time (s) |
|---|---|---|---|---|---|---|---|
| Baseline: AlexNet + BN, lr 1e-3, 15 ep | 15 | 88.17 | 78.54 | +0.00 | 77.99 | 9.63 | 10.77 |
| lr 1e-4, 15 ep | 15 | 96.13 | 81.54 | +3.00 | 81.45 | 14.59 | 15.06 |
| lr 1e-4, early stopping (max 30 ep, patience 5) | 17 | 95.36 | 81.86 | +3.32 | 81.62 | 13.50 | 14.06 |
| lr 1e-2, early stopping (max 30 ep, patience 5) | 22 | 73.08 | 70.27 | -8.27 | 70.21 | 2.81 | 13.27 |
