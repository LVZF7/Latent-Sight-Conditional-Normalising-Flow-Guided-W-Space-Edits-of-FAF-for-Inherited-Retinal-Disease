"""
Objective: UMAP for Real and Synthetic Laterality Data using Inception V3 Backbone
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# load inception v3 model
def load_inception_v3_model(device):
    weights = models.Inception_V3_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.inception_v3(weights=weights)
    model.aux_logits = False # Disable auxiliary logits
    model.fc = nn.Identity()  # Remove the final classification layer, exposing 2048-dim features
    model = model.to(device).eval() # Set to evaluation mode
    for param in model.parameters():
        param.requires_grad = False
    return model, preprocess

# dataset - list of paths and transform to get (tensor, path)
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), img_path
    

# list of png files in the folder
def list_image_files(folder):
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} does not exist.")
    return sorted([str(p) for p in folder.iterdir() if p.suffix.lower() == ".png"]) # only png files, finds them based on suffix and adds to a sorted list


# sample real images
def sample_real_images(real_folder, side, num_samples, seed=42):
    side = side.upper()
    paths = list_image_files(real_folder/side)
    if len(paths) == 0: # no images found
        raise ValueError(f"No images found in {real_folder/side}")
    rng = np.random.default_rng(seed)
    if len(paths) > num_samples: # if there are more images than num_samples
        paths = [paths[i] for i in rng.choice(len(paths), size=num_samples, replace=False)] # sample without replacement
    return paths


# sample synthetic images
def sample_synthetic_images(manifest_csv, side, num_samples, min_conf=0.9, seed=42):
    side = side.upper()
    df = pd.read_csv(manifest_csv)
    df = df[(df['lat_pred'] == side) & (df['lat_conf'] >= min_conf)]
    if len(df) == 0: # no images found
        raise ValueError(f"No synthetic images found for side {side} with min_conf {min_conf}")
    rng = np.random.default_rng(seed)
    if len(df) > num_samples: # if there are more images than num_samples
        df = df.iloc[rng.choice(len(df), size=num_samples, replace=False)]
    return df['img_path'].tolist()


# eXtract features using the model
@torch.no_grad()
def extract_features(img_paths, model, preprocess, device, batch_size=64):
    dataset = ImageDataset(img_paths, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    features = []
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        feats = model(imgs).cpu().numpy().astype(np.float32)
        features.append(feats)
    return np.concatenate(features, axis=0)

# PCA
def apply_pca(feature, n_components, seed=42):
    if n_components and n_components > 0:
        pca = PCA(n_components=n_components, random_state=seed)
        return pca.fit_transform(feature)
    return feature

# apply the UMAP reduction
def apply_umap(features, n_components, n_neighbours, min_dist, metric='euclidean', seed=42):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbours, min_dist=min_dist, metric=metric, random_state=seed)
    return reducer.fit_transform(features)

# plot the UMAP embeddings
def plot_umap_2d(embeddings, labels, output_path, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def side(label):
        return 'L' if '-L' in label else 'R' # determine side from label
    def domain(label):
        return 'Real' if label.startswith('Real-') else 'Synthetic' # determine domain from label
    
    colours = {'L': 'blue', 'R': 'red'}
    markers = {'Real': 'o', 'Synthetic': 'x'}
    
    plt.figure(figsize=(8, 8))
    for ul in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == ul]
        plt.scatter(embeddings[idx, 0], embeddings[idx, 1], c=colours[side(ul)], marker=markers[domain(ul)], label=ul, s=18, alpha=0.7)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    
# plot the UMAP embeddings in 3D
def plot_umap_3d(embeddings, labels, output_path, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def side(label):
        return 'L' if '-L' in label else 'R' # determine side from label
    def domain(label):
        return 'Real' if label.startswith('Real-') else 'Synthetic' # determine domain from label
    
    colours = {'L': 'blue', 'R': 'red'}
    markers = {'Real': 'o', 'Synthetic': 'x'}
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for ul in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == ul]
        ax.scatter(embeddings[idx, 0], embeddings[idx, 1], embeddings[idx, 2], c=colours[side(ul)], marker=markers[domain(ul)], label=ul, s=18, alpha=0.7)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_zlabel('UMAP Dimension 3')
    ax.set_title(title)
    ax.legend()
    plt.savefig(output_path)
    plt.close()
    
# save embeddings to CSV    
def save_embeddings(embeddings, labels, paths, output_csv):
    cols = ["x", "y"] if embeddings.shape[1] == 2 else ["x", "y", "z"]
    df = pd.DataFrame(embeddings, columns=cols)
    df['label'] = labels
    df['img_path'] = paths
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_folder", type=Path, required=True)
    p.add_argument("--synthetic_manifest", type=Path, required=True)
    p.add_argument("--output_folder", type=Path, required=True)

    p.add_argument("--num_per_group", type=int, default=500)
    p.add_argument("--min_conf", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--pca_components", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min_dist", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    
    # fix seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # build image lists
    real_L = sample_real_images(args.real_folder, 'L', args.num_per_group, seed=args.seed)
    real_R = sample_real_images(args.real_folder, 'R', args.num_per_group, seed=args.seed)
    synth_L = sample_synthetic_images(args.synthetic_manifest, 'L', args.num_per_group, min_conf=args.min_conf, seed=args.seed)
    synth_R = sample_synthetic_images(args.synthetic_manifest, 'R', args.num_per_group, min_conf=args.min_conf, seed=args.seed)
    
    paths = real_L + real_R + synth_L + synth_R
    labels = (['Real-L'] * len(real_L)) + (['Real-R'] * len(real_R)) + (['Synthetic-L'] * len(synth_L)) + (['Synthetic-R'] * len(synth_R))

    # load backbone model
    device = torch.device(args.device)
    model, preprocess = load_inception_v3_model(device)
    
    # extract features
    features = extract_features(paths, model, preprocess, device, batch_size=args.batch_size)
    
    # normalise and PCA
    features = features / np.linalg.norm(features, axis=1, keepdims=True) # L2 normalisation
    PCA_features = apply_pca(features, args.pca_components, seed=args.seed)
    
    # 2D UMAP
    embeddings_2d = apply_umap(PCA_features, n_components=2, n_neighbours=args.n_neighbors, min_dist=args.min_dist, metric='euclidean', seed=args.seed)
    save_embeddings(embeddings_2d, labels, paths, args.output_folder / "embeddings_inception_2d.csv")
    plot_umap_2d(embeddings_2d, labels, args.output_folder / "umap_inception_2d.png", title="Real vs Synthetic UMAP 2D (Inception V3)")
    
    print(f"Saved 2D UMAP plot and embeddings to {args.output_folder}")
    
    # 3D UMAP
    embeddings_3d = apply_umap(PCA_features, n_components=3, n_neighbours=args.n_neighbors, min_dist=args.min_dist, metric='euclidean', seed=args.seed)
    save_embeddings(embeddings_3d, labels, paths, args.output_folder / "embeddings_inception_3d.csv")
    plot_umap_3d(embeddings_3d, labels, args.output_folder / "umap_inception_3d.png", title="Real vs Synthetic UMAP 3D (Inception V3)")

    print(f"Saved 3D UMAP plot and embeddings to {args.output_folder}")

if __name__ == "__main__":
    main()
    
    
"""
Usage Example:
python scripts/results/Laterality/umap/umap_lat_inceptionBB.py \
  --real_folder data/samples_umap/real \
  --synthetic_manifest data/laterality/generated/manifest_generated.csv \
  --output_folder experiment/laterality/results_umap_inception \
  --num_per_group 1000 \
  --min_conf 0.90 \
  --batch_size 64 \
  --pca_components 64 \
  --n-neighbors 30 \
  --min_dist 0.10 \
  --seed 42 \
  --device cuda
"""