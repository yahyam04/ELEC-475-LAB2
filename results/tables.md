### Table 7.3a: Architectural Configuration & Computational Complexity (CIFAR-10 runs)

| Model Architecture | Conv Depth | Normalization & Regularization | Data Aug | Params (M) | FLOPs (M) | Mean Epoch Time (s) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| 1. Base LeNet-5 | 2 | None (Tanh + AvgPool) | None | 0.338 | 7.17 | 5.44 ± 0.09 | 0.65 |
| 2. Modernized LeNet | 2 | None (ReLU + Overlapping MaxPool) | None | 0.446 | 9.19 | 5.49 ± 0.11 | 0.94 |
| 3. AlexNet (No Norm) | 5 | Dropout (p=0.5) | None | 6.609 | 436.70 | 12.58 ± 5.62 | 1.27 |
| 4. AlexNet + LRN | 5 | LRN + Dropout (p=0.5) | None | 6.609 | 436.70 | 17.73 ± 7.74 | 1.90 |
| 5. AlexNet + BatchNorm | 5 | BatchNorm2d + Dropout (p=0.5) | None | 6.611 | 437.97 | 14.40 ± 7.00 | 1.15 |
| 6. AlexNet + BN + Aug | 5 | BatchNorm2d + Dropout (p=0.5) | Pure PyTorch Aug | 6.611 | 437.97 | 15.84 ± 5.80 | 1.19 |
| 7. PyTorch Zoo AlexNet | 5 | PyTorch Zoo (AdaptiveAvgPool2d) | None | 57.045 | 188.20 | 17.14 ± 7.88 | 1.89 |

### Table 7.3b: Predictive Performance (final-epoch weights; Mean ± Std over seeds)

| Model Architecture | #Seeds (C10 / Tiny) | CIFAR-10 Val Acc (%) | CIFAR-10 Val Macro F1 (%) | Tiny ImageNet Val Acc (%) | Tiny ImageNet Val Macro F1 (%) |
|---|---|---|---|---|---|
| 1. Base LeNet-5 | 3 / 1 | 54.45 ± 0.93 | 54.29 ± 0.93 | 12.34 | 10.48 |
| 2. Modernized LeNet | 3 / 1 | 64.99 ± 0.65 | 64.89 ± 0.68 | 19.72 | 17.45 |
| 3. AlexNet (No Norm) | 3 / 1 | 72.36 ± 1.67 | 71.98 ± 1.78 | 12.92 | 10.32 |
| 4. AlexNet + LRN | 3 / 1 | 76.46 ± 0.36 | 76.32 ± 0.40 | 17.65 | 15.37 |
| 5. AlexNet + BatchNorm | 3 / 1 | 79.47 ± 1.35 | 79.31 ± 1.45 | 29.88 | 28.47 |
| 6. AlexNet + BN + Aug | 3 / 1 | 81.05 ± 0.52 | 80.86 ± 0.60 | 28.92 | 26.96 |
| 7. PyTorch Zoo AlexNet | 3 / 1 | 63.29 ± 4.17 | 63.16 ± 4.39 | 9.30 | 6.63 |

### Generalization Gap Δ = Train Acc − Val Acc (eval mode, no augmentation)

| Dataset | Configuration | Train Acc (%) | Val Acc (%) | Δ (pts) | Val Macro F1 (%) |
|---|---|---|---|---|---|
| cifar10 | 5. AlexNet + BatchNorm | 89.42 ± 1.47 | 79.47 ± 1.35 | 9.95 ± 0.29 | 79.31 ± 1.45 |
| cifar10 | 6. AlexNet + BN + Aug | 84.97 ± 0.81 | 81.05 ± 0.52 | 3.92 ± 0.30 | 80.86 ± 0.60 |
| tiny_imagenet | 5. AlexNet + BatchNorm | 36.42 | 29.88 | 6.54 | 28.47 |
| tiny_imagenet | 6. AlexNet + BN + Aug | 31.46 | 28.92 | 2.54 | 26.96 |
