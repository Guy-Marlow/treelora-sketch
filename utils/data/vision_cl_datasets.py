"""
Split-CIFAR-100, Split-ImageNet-R, and Split-CUB-200 dataloaders for CIL experiments.

Task definitions (matching TreeLoRA paper §5.1):
    Split CIFAR-100:  100 classes  → 10 tasks × 10 classes
    Split ImageNet-R: 200 classes  → 5 / 10 / 20 tasks × 40 / 20 / 10 classes
    Split CUB-200:    200 classes  → 10 tasks × 20 classes

All loaders require datasets to have been downloaded first:
    python utils/data/download_datasets.py --data_root /path/to/data

Each make_* function returns a list of task_info dicts:
    {
        'train':       DataLoader,
        'test':        DataLoader,
        'classes':     list[int],   # global class indices assigned to this task
        'num_classes': int,
    }

Labels inside every DataLoader are remapped to [0, num_classes_per_task).

Normalisation (paper §A.3):
    Images resized to 224×224 and normalised to [0, 1] via ToTensor only.
    No per-channel mean/std subtraction.

Augmentation (train only):
    Natural images (ImageNet-R, CUB-200): RandomResizedCrop(224) + RandomHorizontalFlip.
    CIFAR-100 (32×32 source): Resize(224) + RandomHorizontalFlip.
    RandomResizedCrop on 32×32 with default scale=(0.08,1.0) yields crops as small as
    ~9×9 pixels — effectively noise — so a direct Resize is used instead.
Test:
    Natural images: Resize(256) + CenterCrop(224).
    CIFAR-100: Resize(224) only.
"""

import os
from collections import defaultdict

import numpy as np
import PIL.Image
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Transforms (no per-channel normalisation — paper §A.3: "normalised to [0,1]") ──

# For natural-resolution images (ImageNet-R, CUB-200)
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),          # uint8 → float32 in [0, 1]
])

TEST_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# For CIFAR-100 (32×32 source images — RandomResizedCrop would yield degenerate crops)
CIFAR_TRAIN_TF = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

CIFAR_TEST_TF = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])


# ── Class-split utilities ─────────────────────────────────────────────────────

def _split_classes(num_classes: int, num_tasks: int, seed: int) -> list[list[int]]:
    """
    Randomly shuffle num_classes into num_tasks equal groups.
    Uses a fixed seed for reproducibility.  The last group absorbs any remainder
    when num_classes % num_tasks != 0.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_classes).tolist()
    per_task = num_classes // num_tasks
    groups = [order[i * per_task:(i + 1) * per_task] for i in range(num_tasks - 1)]
    groups.append(order[(num_tasks - 1) * per_task:])
    return groups


# ── HuggingFace image dataset wrapper ────────────────────────────────────────

class HFImageDataset(Dataset):
    """
    PyTorch Dataset wrapping a HuggingFace Arrow dataset for image classification.

    Supports class-filtering and label remapping in a single pass; avoids loading
    all images up front by relying on the Arrow columnar label cache for filtering.

    Parameters
    ----------
    hf_split  : a HuggingFace dataset object (output of load_from_disk()[split])
    img_key   : column name for PIL images
    label_key : column name for integer labels
    indices   : row indices to keep (all rows if None)
    transform : torchvision transform applied to each PIL image
    label_map : dict {global_int_label: local_int_label} or None
    """

    def __init__(self, hf_split, img_key: str, label_key: str,
                 indices=None, transform=None, label_map: dict | None = None):
        self.hf        = hf_split.select(indices) if indices is not None else hf_split
        self.img_key   = img_key
        self.label_key = label_key
        self.transform = transform
        self.label_map = label_map
        # Arrow columnar read — fast even for 50K+ rows
        self._targets  = self.hf[label_key]

    @property
    def targets(self) -> list:
        return self._targets

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, idx: int):
        row = self.hf[idx]
        img = row[self.img_key]
        if not isinstance(img, PIL.Image.Image):
            img = PIL.Image.fromarray(img)
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = row[self.label_key]
        if self.label_map is not None:
            label = self.label_map[label]
        return img, label


# ── Task dataset factory ──────────────────────────────────────────────────────

def _task_dataset(hf_split, img_key: str, label_key: str,
                  class_indices: list, transform) -> HFImageDataset:
    """Filter hf_split to class_indices and remap labels to [0, n)."""
    class_set = set(class_indices)
    all_labels = hf_split[label_key]     # columnar read
    row_indices = [i for i, lbl in enumerate(all_labels) if lbl in class_set]
    label_map   = {c: i for i, c in enumerate(class_indices)}
    return HFImageDataset(hf_split, img_key, label_key,
                          indices=row_indices, transform=transform,
                          label_map=label_map)


def _make_loaders(train_ds, test_ds, batch_size: int, num_workers: int):
    """Wrap two datasets in DataLoaders with standard settings."""
    kw = dict(batch_size=batch_size, num_workers=num_workers,
               pin_memory=True, persistent_workers=(num_workers > 0))
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(test_ds,  shuffle=False, **kw))


# ── Public API ────────────────────────────────────────────────────────────────

def make_split_cifar100(
    data_root: str,
    num_tasks: int = 10,
    batch_size: int = 192,
    seed: int = 42,
    num_workers: int = 4,
) -> list[dict]:
    """
    Split CIFAR-100 into 10 tasks of 10 classes each (paper §5.1).

    Requires download_datasets.py to have been run first.

    Args:
        data_root:   Parent directory passed to download_datasets.py.
        num_tasks:   Number of tasks (default 10 → 10 classes/task).
        batch_size:  Samples per batch (paper: 192).
        seed:        Class-order seed (default 42).
        num_workers: DataLoader worker count.
    """
    path = os.path.join(data_root, 'cifar100')
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f'{path} not found. Run: python utils/data/download_datasets.py '
            f'--data_root {data_root} --datasets cifar100'
        )
    ds = load_from_disk(path)
    img_key, label_key = 'img', 'fine_label'

    splits = _split_classes(100, num_tasks, seed)

    task_info = []
    for class_indices in splits:
        train_ds = _task_dataset(ds['train'], img_key, label_key, class_indices, CIFAR_TRAIN_TF)
        test_ds  = _task_dataset(ds['test'],  img_key, label_key, class_indices, CIFAR_TEST_TF)
        train_loader, test_loader = _make_loaders(train_ds, test_ds, batch_size, num_workers)
        task_info.append({
            'train':       train_loader,
            'test':        test_loader,
            'classes':     class_indices,
            'num_classes': len(class_indices),
        })
    return task_info


def make_split_imagenet_r(
    data_root: str,
    num_tasks: int = 10,
    batch_size: int = 192,
    seed: int = 42,
    num_workers: int = 4,
) -> list[dict]:
    """
    Split ImageNet-R into 5, 10, or 20 tasks (paper §5.1).

    200 classes → 5 tasks × 40 classes, 10 tasks × 20 classes, or 20 tasks × 10 classes.
    Requires download_datasets.py to have been run first (which creates the int_label
    column and the 80/20 train/test split).

    Args:
        data_root: Parent directory passed to download_datasets.py.
        num_tasks: 5, 10, or 20 (paper evaluates all three).
        batch_size: Samples per batch (paper: 192).
        seed:      Class-order seed (default 42).
        num_workers: DataLoader worker count.
    """
    if num_tasks not in (5, 10, 20):
        raise ValueError(f'num_tasks must be 5, 10, or 20 for ImageNet-R (got {num_tasks})')

    path = os.path.join(data_root, 'imagenet_r')
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f'{path} not found. Run: python utils/data/download_datasets.py '
            f'--data_root {data_root} --datasets imagenet_r'
        )
    ds = load_from_disk(path)
    img_key, label_key = 'image', 'int_label'

    splits = _split_classes(200, num_tasks, seed)

    task_info = []
    for class_indices in splits:
        train_ds = _task_dataset(ds['train'], img_key, label_key, class_indices, TRAIN_TF)
        test_ds  = _task_dataset(ds['test'],  img_key, label_key, class_indices, TEST_TF)
        train_loader, test_loader = _make_loaders(train_ds, test_ds, batch_size, num_workers)
        task_info.append({
            'train':       train_loader,
            'test':        test_loader,
            'classes':     class_indices,
            'num_classes': len(class_indices),
        })
    return task_info


def make_split_cub200(
    data_root: str,
    num_tasks: int = 10,
    batch_size: int = 192,
    seed: int = 42,
    num_workers: int = 4,
) -> list[dict]:
    """
    Split CUB-200-2011 into 10 tasks of 20 classes each (paper §5.1).

    Requires download_datasets.py to have been run first.

    Args:
        data_root:   Parent directory passed to download_datasets.py.
        num_tasks:   Number of tasks (default 10 → 20 classes/task).
        batch_size:  Samples per batch (paper: 192).
        seed:        Class-order seed (default 42).
        num_workers: DataLoader worker count.
    """
    path = os.path.join(data_root, 'cub200')
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f'{path} not found. Run: python utils/data/download_datasets.py '
            f'--data_root {data_root} --datasets cub200'
        )
    ds = load_from_disk(path)
    img_key, label_key = 'image', 'label'

    splits = _split_classes(200, num_tasks, seed)

    task_info = []
    for class_indices in splits:
        train_ds = _task_dataset(ds['train'], img_key, label_key, class_indices, TRAIN_TF)
        test_ds  = _task_dataset(ds['test'],  img_key, label_key, class_indices, TEST_TF)
        train_loader, test_loader = _make_loaders(train_ds, test_ds, batch_size, num_workers)
        task_info.append({
            'train':       train_loader,
            'test':        test_loader,
            'classes':     class_indices,
            'num_classes': len(class_indices),
        })
    return task_info