import argparse
import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Map your class indices to CSV gene names
CLASS_MAP = {
    0: "ABCA4",
    35: "USH2A",
}

# Sort and copy real images based on gene classes from a CSV file
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--classes", type=str, default="0,35")   # e.g. "0,35"
    p.add_argument("--per_class", type=int, default=2000)
    p.add_argument("--out_root", type=Path, default=Path("data/samples_umap/real"))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df["gene"] = df["gene"].astype(str)
    df["file_path_original"] = df["file_path_original"].astype(str)

    os.makedirs(args.out_root, exist_ok=True)

    for token in [s.strip() for s in args.classes.split(",") if s.strip()]:
        idx = int(token)
        gene_name = CLASS_MAP.get(idx, str(idx))

        dest_dir = args.out_root / f"gene_{idx}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        paths = df[df["gene"].str.upper() == gene_name.upper()]["file_path_original"].tolist()

        copied = 0
        for src in tqdm(paths, total=min(len(paths), args.per_class), desc=f"gene_{idx}", leave=False):
            if copied >= args.per_class:
                break
            if os.path.exists(src):
                dst = dest_dir / f"gene{idx}_{copied:04d}_{Path(src).name}"
                shutil.copy2(src, dst)
                copied += 1

    print("done")

if __name__ == "__main__":
    main()


"""
Usage:
python scripts/results/All/umap_scripts/sort_gene_reals_umap.py \
  --csv /mnt/data/ajohn/multiconditioning/StyleFlow_Based_Edit/data/real/nnunet_faf_v0_dataset_v2_local.csv \
  --classes 0,35 \
  --per_class 2000 \
  --out_root data/samples_umap/real
"""