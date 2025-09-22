import argparse, os, glob
import numpy as np
import pandas as pd
from pathlib import Path

# load W latent from npz file
def load_w(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"W latent file not found: {npz_path}")
    
    d = np.load(npz_path)
    
    # Try different possible keys for W latents
    for k in ['w', 'W', 'latents', 'dlatent']:
        if k in d:
            arr = d[k]
            break
    else:
        available_keys = list(d.keys())
        raise KeyError(f"No W-like key found in {npz_path}. Available keys: {available_keys}")
    
    # Handle different W shapes
    if arr.ndim == 3: # [1, 18, 512] -> [512]
        arr = arr[0].mean(axis=0)
    elif arr.ndim == 2: # [1, 512] -> [512]
        arr = arr[0]
    elif arr.ndim == 1: # [512] -> [512]
        pass
    else:
        raise ValueError(f"Unexpected W shape: {arr.shape}")
    
    return arr.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests', nargs='+')
    parser.add_argument('--generated_dir', default='data/gene/generated')
    parser.add_argument('--outdir', default='data/gene/numpy')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--per_class_max', type=int, default=200)
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Auto-discover manifests if not provided
    if not args.manifests:
        manifest_pattern = os.path.join(args.generated_dir, "gene_*", "manifest_generated.csv")
        args.manifests = glob.glob(manifest_pattern)
        if not args.manifests:
            raise FileNotFoundError(f"No manifests found in {manifest_pattern}")
    
    print(f"Found {len(args.manifests)} manifest files:")
    for m in args.manifests:
        print(f"  - {m}")
    print()

    # Load all manifests
    all_rows = []
    for manifest_path in args.manifests:
        print(f"Loading {manifest_path}...")
        df = pd.read_csv(manifest_path)
        
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")

        # Adapt to your manifest column names
        if 'gene_class_idx' in df.columns:
            df['class'] = df['gene_class_idx']
        else:
            raise ValueError(f"Expected 'gene_class_idx' column in {manifest_path}")
        
        # Map column names to standard names
        if 'img_path' in df.columns:
            df['image_path'] = df['img_path'] 
        else:
            raise ValueError(f"Expected 'img_path' column in {manifest_path}")
            
        if 'w_npz_path' in df.columns:
            df['w_path'] = df['w_npz_path']
        else:
            raise ValueError(f"Expected 'w_npz_path' column in {manifest_path}")
        
        # Check if paths are already absolute/full
        sample_w_path = df['w_path'].iloc[0]
        if os.path.isabs(sample_w_path) or sample_w_path.startswith('data/'):
            # Paths are already full - use as-is
            df['w_path_full'] = df['w_path']
            print(f"  Using full paths as-is: {sample_w_path}")
        else:
            # Paths are relative to manifest directory
            manifest_dir = os.path.dirname(manifest_path)
            df['w_path_full'] = df['w_path'].apply(lambda x: os.path.join(manifest_dir, x))
            print(f"  Making paths relative to: {manifest_dir}")
        
        # Select standardized columns
        all_rows.append(df[['image_path', 'w_path_full', 'class']])
    
    df = pd.concat(all_rows, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    print(f"Class distribution: {df['class'].value_counts().sort_index()}")

    # Build arrays
    latents, attrs = [], []
    print(f"\nBuilding arrays (max {args.per_class_max} per class)...")
    
    for c in range(args.num_classes):
        class_rows = df[df['class'] == c].head(args.per_class_max)
        print(f"Gene class {c}: {len(class_rows)} samples")

        for _, row in class_rows.iterrows():
            try:
                w = load_w(row['w_path_full'])
                
                # Create one-hot encoding for gene class
                onehot = np.zeros(args.num_classes, dtype=np.float32)
                onehot[c] = 1.0
                
                latents.append(w)
                attrs.append(onehot)
                
            except Exception as e:
                print(f"Warning: Failed to load {row['w_path_full']}: {e}")
                continue

    if len(latents) == 0:
        raise ValueError("No latents were successfully loaded!")

    # Save arrays
    latents_array = np.stack(latents)
    attrs_array = np.stack(attrs)
    
    latents_path = os.path.join(args.outdir, 'latents.npy')
    attrs_path = os.path.join(args.outdir, 'attributes.npy')
    
    np.save(latents_path, latents_array)
    np.save(attrs_path, attrs_array)
    
    print(f"\nArrays saved:")
    print(f"- Latents: {latents_path} {latents_array.shape}")
    print(f"- Attributes: {attrs_path} {attrs_array.shape}")
    print(f"- Total samples: {len(latents)}")

    # Verify class distribution
    class_dist = attrs_array.argmax(1)
    unique, counts = np.unique(class_dist, return_counts=True)
    print(f"- Final class distribution: {dict(zip(unique, counts))}")

if __name__ == '__main__':
    main()