# Used to remap paths in a CSV file from NAS paths to local paths. I've used this to prep for umap visualisations.

import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', default='data/real/nnunet_faf_v0_dataset_v2.csv')
    parser.add_argument('--output_csv', default='data/real/nnunet_faf_v0_dataset_v2_local.csv')
    parser.add_argument('--local_base', default='/mnt/data/ajohn/multiconditioning/StyleFlow_Based_Edit/data/real/real_images')
    args = parser.parse_args()
    
    # Read CSV
    df = pd.read_csv(args.input_csv)
    
    # Remap paths
    old_base = '/media/pontikos_nas2/NikolasPontikos/IRD/new_export'
    
    for col in df.columns:
        if 'path' in col.lower():
            df[col] = df[col].str.replace(old_base, args.local_base, regex=False)
    
    # Save new CSV
    df.to_csv(args.output_csv, index=False)
    
    print(f"Remapped {len(df)} rows")
    print(f"Saved to: {args.output_csv}")

if __name__ == '__main__':
    main()