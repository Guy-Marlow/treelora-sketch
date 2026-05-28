"""
Download ViT benchmark datasets from HuggingFace and save to local disk.

Datasets
--------
    cifar100    uoft-cs/cifar100           img / fine_label   train + test
    imagenet_r  axiong/imagenet-r          image / wnid       test only (we split 80/20)
    cub200      Donghyun99/CUB-200-2011    image / label      train + test

Run once before training:

    conda run -n treelora python utils/data/download_datasets.py \\
        --data_root /path/to/data

    # or just one dataset:
        --datasets imagenet_r

Each dataset is saved under {data_root}/{name}/ in HuggingFace Arrow format.
The training loaders read from disk and require no internet connection.

ImageNet-R note
---------------
The source dataset (axiong/imagenet-r) has only a 'test' split and uses WordNet
IDs (wnid) rather than integer labels.  The download step adds an 'int_label'
column (0-199, sorted by wnid) and performs a stratified 80/20 train/val split,
saving the result as separate 'train' and 'test' DatasetDict entries on disk.
"""

import argparse
import json
import os

from datasets import Dataset, DatasetDict, load_dataset


# ── Dataset descriptors ───────────────────────────────────────────────────────

DATASET_CONFIGS = {
    'cifar100': {
        'hf_name':   'uoft-cs/cifar100',
        'hf_splits': ['train', 'test'],
        'img_key':   'img',
        'label_key': 'fine_label',
        'num_classes': 100,
    },
    'imagenet_r': {
        'hf_name':   'axiong/imagenet-r',
        'hf_splits': ['test'],   # only split available; we create train/test below
        'img_key':   'image',
        'label_key': 'int_label',  # computed from wnid during download
        'num_classes': 200,
    },
    'cub200': {
        'hf_name':   'Donghyun99/CUB-200-2011',
        'hf_splits': ['train', 'test'],
        'img_key':   'image',
        'label_key': 'label',
        'num_classes': 200,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stratified_split(dataset, label_col: str, val_frac: float = 0.2, seed: int = 42):
    """
    Split a HuggingFace dataset into train/val using stratified sampling.
    Returns (train_dataset, val_dataset).
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    labels = dataset[label_col]
    from collections import defaultdict
    by_class = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_class[lbl].append(i)

    train_idx, val_idx = [], []
    for cls_indices in by_class.values():
        shuffled = rng.permutation(cls_indices).tolist()
        n_val = max(1, int(len(shuffled) * val_frac))
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    return dataset.select(train_idx), dataset.select(val_idx)


def _add_int_labels(dataset, wnid_col: str = 'wnid') -> tuple:
    """
    Add an 'int_label' column derived from sorted unique wnids (0-indexed).
    Returns (updated_dataset, wnid_to_int mapping dict).
    """
    wnids = sorted(set(dataset[wnid_col]))
    wnid_to_int = {w: i for i, w in enumerate(wnids)}
    int_labels = [wnid_to_int[w] for w in dataset[wnid_col]]
    return dataset.add_column('int_label', int_labels), wnid_to_int


# ── Per-dataset download logic ────────────────────────────────────────────────

def _download_cifar100(save_path: str):
    print('  Downloading uoft-cs/cifar100 …')
    ds = load_dataset('uoft-cs/cifar100')
    ds.save_to_disk(save_path)
    print(f'  Saved → {save_path}')
    print(f'    train: {len(ds["train"]):,}  test: {len(ds["test"]):,}')


def _download_imagenet_r(save_path: str, val_frac: float = 0.2, seed: int = 42):
    print('  Downloading axiong/imagenet-r …')
    full = load_dataset('axiong/imagenet-r', split='test')

    print(f'  Total images: {len(full):,}')
    print('  Adding integer labels from wnids …')
    full, wnid_map = _add_int_labels(full, wnid_col='wnid')

    print('  Creating stratified 80/20 train/val split …')
    train_ds, test_ds = _stratified_split(full, label_col='int_label',
                                          val_frac=val_frac, seed=seed)

    ds = DatasetDict({'train': train_ds, 'test': test_ds})
    ds.save_to_disk(save_path)

    # Save wnid→int mapping alongside the dataset
    with open(os.path.join(save_path, 'wnid_to_int.json'), 'w') as f:
        json.dump(wnid_map, f)

    print(f'  Saved → {save_path}')
    print(f'    train: {len(train_ds):,}  test: {len(test_ds):,}  classes: {len(wnid_map)}')


def _download_cub200(save_path: str):
    print('  Downloading Donghyun99/CUB-200-2011 …')
    ds = load_dataset('Donghyun99/CUB-200-2011')
    ds.save_to_disk(save_path)
    print(f'  Saved → {save_path}')
    print(f'    train: {len(ds["train"]):,}  test: {len(ds["test"]):,}')


_DOWNLOADERS = {
    'cifar100':   _download_cifar100,
    'imagenet_r': _download_imagenet_r,
    'cub200':     _download_cub200,
}


# ── Entry point ───────────────────────────────────────────────────────────────

def download(name: str, data_root: str, force: bool = False, **kwargs):
    save_path = os.path.join(data_root, name)
    if os.path.exists(save_path) and not force:
        print(f'  {name}: already present at {save_path} (use --force to re-download)')
        return
    _DOWNLOADERS[name](save_path, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description='Download ViT benchmark datasets from HuggingFace'
    )
    parser.add_argument('--data_root', required=True,
                        help='Parent directory; each dataset goes in {data_root}/{name}/')
    parser.add_argument('--datasets', nargs='+',
                        default=list(DATASET_CONFIGS.keys()),
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Which datasets to download (default: all)')
    parser.add_argument('--force', action='store_true',
                        help='Re-download even if the directory already exists')
    parser.add_argument('--imagenet_r_val_frac', type=float, default=0.2,
                        help='Validation fraction for ImageNet-R (default 0.2)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)

    for name in args.datasets:
        print(f'\n[{name}]')
        kw = {}
        if name == 'imagenet_r':
            kw = {'val_frac': args.imagenet_r_val_frac, 'seed': args.seed}
        download(name, args.data_root, force=args.force, **kw)

    print('\nAll done. Saved metadata per dataset:')
    for name in args.datasets:
        cfg = DATASET_CONFIGS[name]
        print(f'  {name}: img_key={cfg["img_key"]}  label_key={cfg["label_key"]}'
              f'  num_classes={cfg["num_classes"]}')


if __name__ == '__main__':
    main()
