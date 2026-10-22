"""
ELEC 475: Lab 2 - Unified Dataset & DataLoader Factory
Supports CIFAR-10 (primary) and Tiny ImageNet-200 (secondary benchmark).
"""
import os
import urllib.request
import zipfile
from typing import Optional
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, ImageFolder
from .transforms import get_train_transforms, get_test_transforms
from .seed import seed_worker

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def download_and_extract_tiny_imagenet(data_dir: str = "./data"):
    target_dir = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.exists(target_dir):
        return target_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading Tiny ImageNet-200 from {TINY_IMAGENET_URL}...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("Download complete.")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.")
    return target_dir

class TinyImageNetVal(Dataset):
    """
    Official Tiny ImageNet validation split (10,000 images, 50 per class).

    Images live flat in  <root>/val/images/  and labels in  <root>/val/val_annotations.txt
    (tab separated: filename, wnid, bbox...). Labels are mapped through the *training* set's
    class_to_idx so that class indices match the training labels exactly.
    """
    def __init__(self, root: str, class_to_idx: dict, transform=None):
        self.transform = transform
        img_dir = os.path.join(root, "val", "images")
        ann_file = os.path.join(root, "val", "val_annotations.txt")
        self.samples = []
        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                fname, wnid = parts[0], parts[1]
                self.samples.append((os.path.join(img_dir, fname), class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_train_eval_loader(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    target_size: int = 64
) -> DataLoader:
    """
    Training split passed through the deterministic *test* transforms (no augmentation, no shuffling).
    Evaluating the model in eval() mode on this loader gives the clean training accuracy needed for
    the generalization gap  Delta = Train Acc - Val Acc  (the running train accuracy logged during
    training is measured in train() mode, with dropout active and, if enabled, on augmented images).
    """
    name = dataset.lower()
    val_tf = get_test_transforms("tiny_imagenet" if name != "cifar10" else "cifar10", target_size=target_size)
    if name == "cifar10":
        ds = CIFAR10(root=data_dir, train=True, download=True, transform=val_tf)
    else:
        tiny_dir = download_and_extract_tiny_imagenet(data_dir)
        ds = ImageFolder(root=os.path.join(tiny_dir, "train"), transform=val_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=pin_memory and torch.cuda.is_available(), drop_last=False)


def get_dataloaders(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_augmentation: bool = True,
    target_size: int = 64,
    seed: Optional[int] = None
):
    dataset_name = dataset.lower()

    if dataset_name == "cifar10":
        train_tf = get_train_transforms("cifar10", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("cifar10", target_size=target_size)

        train_set = CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
        val_set   = CIFAR10(root=data_dir, train=False, download=True, transform=val_tf)
        num_classes = 10
        classes = CIFAR10_CLASSES

    elif dataset_name in ["tiny_imagenet", "tinyimagenet"]:
        tiny_dir = download_and_extract_tiny_imagenet(data_dir)
        train_dir = os.path.join(tiny_dir, "train")

        train_tf = get_train_transforms("tiny_imagenet", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("tiny_imagenet", target_size=target_size)

        train_set = ImageFolder(root=train_dir, transform=train_tf)
        # BUGFIX: the starter code built the "validation" set from the *training* folder, so the
        # reported Tiny ImageNet validation accuracy was measured on images the model trained on.
        # Use the official held-out val/ split instead.
        val_set   = TinyImageNetVal(tiny_dir, train_set.class_to_idx, transform=val_tf)
        num_classes = 200
        classes = train_set.classes

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from 'cifar10', 'tiny_imagenet'.")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker if seed is not None else None,
        generator=generator
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )

    return train_loader, val_loader, num_classes, classes
