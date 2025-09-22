import os, argparse
import pandas as pd
import numpy as np

# Create balanced flow arrays for laterality (L/R) with per-class confidence thresholds
# Saves latents.npy and attributes.npy in outdir
# Attributes are one-hot: L=[1,0], R=[0,1]
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--min_conf_l', type=float, default=0.75)
    p.add_argument('--min_conf_r', type=float, default=0.90)
    p.add_argument('--use_wplus', type=int, default=0)
    p.add_argument('--per_class_max', type=int, default=500)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.manifest)

    print(f"Original dataset: {len(df)} samples")
    
    # Apply different thresholds per class
    l_samples = df[(df['lat_pred'] == 'L') & (df['lat_conf'] >= args.min_conf_l)].copy()
    r_samples = df[(df['lat_pred'] == 'R') & (df['lat_conf'] >= args.min_conf_r)].copy()
    
    print(f"After per-class filtering (L>={args.min_conf_l}, R>={args.min_conf_r}):")
    print(f"  L samples: {len(l_samples)}")
    print(f"  R samples: {len(r_samples)}")
    
    if len(l_samples) == 0 or len(r_samples) == 0:
        print("Cannot create balanced dataset - one class has no samples")
        print("Try lowering confidence thresholds")
        return
    
    # Sort by confidence (descending) and take top samples
    l_samples = l_samples.nlargest(min(args.per_class_max, len(l_samples)), 'lat_conf')
    r_samples = r_samples.nlargest(min(args.per_class_max, len(r_samples)), 'lat_conf')
    
    # Balance classes (take min count from each)
    n_per_class = min(len(l_samples), len(r_samples))
    l_samples = l_samples.head(n_per_class)
    r_samples = r_samples.head(n_per_class)
    
    print(f"Balanced selection: {n_per_class} samples per class ({2*n_per_class} total)")
    
    # Combine and shuffle
    balanced_df = pd.concat([l_samples, r_samples]).sample(frac=1).reset_index(drop=True)
    
    latents = []
    attrs = []
    
    for _, row in balanced_df.iterrows():
        # Load W codes
        npz = np.load(row['w_npz_path'])
        if args.use_wplus:
            ws = npz['ws']        # (num_ws, 512)
            w = ws.mean(axis=0)   # (512,)
        else:
            w = npz['w']          # (512,)
        latents.append(w.astype(np.float32))
        
        # Convert laterality to one-hot
        lat = row['lat_pred']
        a = np.array([1,0], dtype=np.float32) if lat == 'L' else np.array([0,1], dtype=np.float32)
        attrs.append(a)
    
    latents = np.stack(latents, axis=0)  # (N,512)
    attrs = np.stack(attrs, axis=0)      # (N,2)
    
    # Save arrays
    latents_path = os.path.join(args.outdir, 'latents.npy')
    attrs_path = os.path.join(args.outdir, 'attributes.npy')
    
    np.save(latents_path, latents)
    np.save(attrs_path, attrs)
    
    print(f"Saved: {latents.shape} -> {latents_path}")
    print(f"Saved: {attrs.shape} -> {attrs_path}")
    
    # Print confidence stats
    l_conf_avg = l_samples['lat_conf'].mean()
    r_conf_avg = r_samples['lat_conf'].mean()
    l_conf_min = l_samples['lat_conf'].min()
    r_conf_min = r_samples['lat_conf'].min()
    
    print(f"Final confidence stats:")
    print(f"  L: avg={l_conf_avg:.3f}, min={l_conf_min:.3f}")
    print(f"  R: avg={r_conf_avg:.3f}, min={r_conf_min:.3f}")

if __name__ == '__main__':
    main()