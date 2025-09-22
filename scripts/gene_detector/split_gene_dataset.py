import argparse
import random
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

GENE_TO_IDX = {"ABCA4": 0, "USH2A": 35} # Only these two classes for gene detector

# Stratified split into train/val/test
def stratified_split(labels, train_size=0.8, val_size=0.1, test_size=0.1, seed=42):
    total = train_size + val_size + test_size
    train_size, val_size, test_size = train_size/total, val_size/total, test_size/total

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1-train_size, random_state=seed)
    train_idx, valtest_idx = next(splitter.split(np.zeros((len(labels), 1)), labels))

    valtest_labels = labels[valtest_idx]
    val_ratio = val_size / (val_size + test_size)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1-val_ratio, random_state=seed)
    val_idx, test_idx = next(splitter.split(np.zeros((len(valtest_labels), 1)), valtest_labels))

    n = len(labels)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[train_idx] = True
    val_mask[valtest_idx[val_idx]] = True
    test_mask[valtest_idx[test_idx]] = True
    return train_mask, val_mask, test_mask

# Copy files to split directories
def copy_split(df_split, dest_root, split_name):
    split_dir = dest_root / split_name
    dest_cls0 = split_dir / "gene_0"
    dest_cls35 = split_dir / "gene_35"
    dest_cls0.mkdir(parents=True, exist_ok=True)
    dest_cls35.mkdir(parents=True, exist_ok=True)

    counters = {0: 0, 35: 0}
    for row in df_split.itertuples(index=False):
        image_path = Path(row.file_path_original)
        cls_idx = int(row.cls_idx)
        if not image_path.is_file():
            continue

        dest_dir = dest_cls0 if cls_idx == 0 else dest_cls35 if cls_idx == 35 else None
        if dest_dir is None:
            continue

        idx = counters[cls_idx]
        dest_path = dest_dir / f"gene_{cls_idx}_{idx}_{image_path.name}"
        while dest_path.exists():
            idx += 1
            dest_path = dest_dir / f"gene_{cls_idx}_{idx}_{image_path.name}"

        shutil.copy(image_path, dest_path)
        counters[cls_idx] = idx + 1

# Extract dataset from CSV and create splits
def extract_from_csv(csv_path, output_dir, seed):
    df = pd.read_csv(csv_path)
    df = df[df['gene'].isin(GENE_TO_IDX.keys())].copy()
    df["cls_idx"] = df["gene"].map(GENE_TO_IDX)

    labels = df["cls_idx"].to_numpy()
    train_mask, val_mask, test_mask = stratified_split(labels, 0.8, 0.1, 0.1, seed)

    copy_split(df[train_mask], output_dir, "train")
    copy_split(df[val_mask], output_dir, "val")
    copy_split(df[test_mask], output_dir, "test")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)  # e.g., data/real/geneCNN_training_splits
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    extract_from_csv(args.csv_path, args.output_dir, args.seed)

if __name__ == "__main__":
    main()
    
"""
Usage:
python scripts/gene_detector/split_gene_dataset.py \
  --csv_path data/real/nnunet_faf_v0_dataset_v2_local.csv \
  --output_dir data/real/geneCNN_training_splits \
  --seed 42
"""