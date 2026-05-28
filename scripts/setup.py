"""
One-command setup for the TreeLoRA ViT continual-learning project.

Downloads the ViT-B/16 iBOT-21K checkpoint and any requested datasets from
HuggingFace.  All steps are idempotent — if a directory already exists and
looks complete the step is skipped entirely.

Usage (run from the treelora/ directory):

    conda run -n treelora python scripts/setup.py

Options:
    --ptm_dir   PTM/vit-base-patch16-224-in21k   where to save the checkpoint
    --data_root data                              parent dir for datasets
    --datasets  cifar100 imagenet_r cub200        which datasets to download
    --force                                       re-download even if present
"""

import argparse
import os
import sys


# ── Helpers ───────────────────────────────────────────────────────────────────

def _looks_complete(path: str, required_files: list[str]) -> bool:
    """True iff path exists and contains all required_files."""
    if not os.path.isdir(path):
        return False
    return all(os.path.exists(os.path.join(path, f)) for f in required_files)


def _section(title: str):
    print(f'\n{"─" * 60}')
    print(f'  {title}')
    print(f'{"─" * 60}')


# ── Checkpoint download ───────────────────────────────────────────────────────

_PTM_HF_ID      = 'google/vit-base-patch16-224-in21k'
_PTM_SENTINELS  = ['config.json', 'model.safetensors']   # files that must exist


def setup_ptm(ptm_dir: str, force: bool = False):
    _section(f'ViT-B/16 checkpoint  →  {ptm_dir}')

    if not force and _looks_complete(ptm_dir, _PTM_SENTINELS):
        print(f'  Already present — skipping.')
        return

    from huggingface_hub import snapshot_download

    print(f'  Downloading {_PTM_HF_ID} from HuggingFace Hub …')
    os.makedirs(ptm_dir, exist_ok=True)
    snapshot_download(
        repo_id=_PTM_HF_ID,
        repo_type='model',
        local_dir=ptm_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=['*.msgpack', 'tf_model*', 'flax_model*', 'rust_model*'],
    )
    print(f'  Saved → {ptm_dir}')


# ── Dataset downloads ─────────────────────────────────────────────────────────

_DATASET_SENTINELS = {
    'cifar100':   ['dataset_dict.json', 'train', 'test'],
    'imagenet_r': ['dataset_dict.json', 'train', 'test', 'wnid_to_int.json'],
    'cub200':     ['dataset_dict.json', 'train', 'test'],
}


def setup_datasets(data_root: str, datasets: list[str], force: bool = False):
    _section(f'Datasets  →  {data_root}')

    import subprocess
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dl_script  = os.path.join(script_dir, 'utils', 'data', 'download_datasets.py')

    for name in datasets:
        path = os.path.join(data_root, name)
        sentinels = _DATASET_SENTINELS.get(name, [])
        if not force and _looks_complete(path, sentinels):
            print(f'  {name}: already present — skipping.')
            continue
        print(f'\n  [{name}]')
        cmd = [sys.executable, dl_script, '--data_root', data_root, '--datasets', name]
        if force:
            cmd.append('--force')
        subprocess.run(cmd, check=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Download TreeLoRA checkpoint and datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--ptm_dir', default='PTM/vit-base-patch16-224-in21k',
        help='Destination for the ViT-B/16 iBOT-21K checkpoint',
    )
    p.add_argument(
        '--data_root', default='data',
        help='Parent directory for datasets (one sub-folder per dataset)',
    )
    p.add_argument(
        '--datasets', nargs='+',
        default=['cifar100', 'imagenet_r', 'cub200'],
        choices=['cifar100', 'imagenet_r', 'cub200'],
        help='Which datasets to download',
    )
    p.add_argument(
        '--force', action='store_true',
        help='Re-download even if already present',
    )
    p.add_argument(
        '--skip_ptm', action='store_true',
        help='Skip the ViT checkpoint download (datasets only)',
    )
    p.add_argument(
        '--skip_data', action='store_true',
        help='Skip dataset downloads (checkpoint only)',
    )
    args = p.parse_args()

    if not args.skip_ptm:
        setup_ptm(args.ptm_dir, force=args.force)

    if not args.skip_data:
        os.makedirs(args.data_root, exist_ok=True)
        setup_datasets(args.data_root, args.datasets, force=args.force)

    print(f'\n{"─" * 60}')
    print('  Setup complete.')
    print(f'{"─" * 60}')


if __name__ == '__main__':
    main()
