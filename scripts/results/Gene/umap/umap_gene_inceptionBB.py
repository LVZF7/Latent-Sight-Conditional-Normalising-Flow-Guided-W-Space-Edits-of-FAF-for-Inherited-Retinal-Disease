"""
Objective: UMAP for Real and Synthetic Gene Data using Inception V3 Backbone
"""

import argparse
from pathlib import Path
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


# load inception v3 model (T2 custom)
def load_inception_v3_model(device):
    weights = models.Inception_V3_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.inception_v3(weights=weights)
    model.aux_logits = False
    model.fc = nn.Identity()          
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
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
    return sorted([str(p) for p in folder.iterdir() if p.suffix.lower() == ".png"])


# sample real images 
def sample_real_images(real_root: Path, gene_idx: int, max_n: int, seed: int = 42):
    paths = list_image_files(real_root / f"gene_{gene_idx}")
    if not paths:
        raise ValueError(f"No images found in {real_root / f'gene_{gene_idx}'}")
    rng = np.random.default_rng(seed)
    if len(paths) > max_n:
        paths = [paths[i] for i in rng.choice(len(paths), size=max_n, replace=False)]
    return paths


# sample synthetic images 
def sample_synthetic_images(generated_root: Path, gene_idx: int, max_n: int, seed: int = 42):
    manifest = generated_root / f"gene_{gene_idx}" / "manifest_generated.csv"
    df = pd.read_csv(manifest)
    paths = df["img_path"].astype(str).tolist()
    if not paths:
        raise ValueError(f"No images found in manifest {manifest}")
    rng = np.random.default_rng(seed)
    if len(paths) > max_n:
        paths = [paths[i] for i in rng.choice(len(paths), size=max_n, replace=False)]
    return paths


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
    uniq_classes = sorted({lab.split("-")[1] for lab in labels})  # e.g., "gene0", "gene35"
    
    colours = {gc: ('blue' if gc == 'gene0' else 'red' if gc == 'gene35' else 'gray') for gc in uniq_classes}
    markers = {"Real": "o", "Synthetic": "x"}
    
    plt.figure(figsize=(8, 8))
    for ul in sorted(set(labels)):
        domain, gene_tag = ul.split("-")
        idx = [i for i, l in enumerate(labels) if l == ul]
        plt.scatter(embeddings[idx, 0], embeddings[idx, 1], c=colours[gene_tag], marker=markers[domain], label=ul, s=18, alpha=0.7)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    
# plot the UMAP embeddings in 3D 
def plot_umap_3d(embeddings, labels, output_path, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    uniq_classes = sorted({lab.split("-")[1] for lab in labels})  # e.g., "gene0", "gene35"
    
    colours = {gc: ('blue' if gc == 'gene0' else 'red' if gc == 'gene35' else 'gray') for gc in uniq_classes}
    markers = {"Real": "o", "Synthetic": "x"}
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for ul in sorted(set(labels)):
        domain, gene_tag = ul.split("-")
        idx = [i for i, l in enumerate(labels) if l == ul]
        ax.scatter(embeddings[idx, 0], embeddings[idx, 1], embeddings[idx, 2], c=colours[gene_tag], marker=markers[domain], label=ul, s=18, alpha=0.7)
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
    df["label"] = labels
    df["img_path"] = paths
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_folder", type=Path, required=True)
    p.add_argument("--generated_folder", type=Path, required=True)
    p.add_argument("--output_folder", type=Path, required=True)

    p.add_argument("--classes", type=str, required=True)
    p.add_argument("--num_per_class", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--pca_components", type=int, default=50)
    p.add_argument("--n_neighbors", type=int, default=30)
    p.add_argument("--min_dist", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    
    # fix seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # get gene classes
    gene_indices = [int(s.strip()) for s in args.classes.split(",") if s.strip() != ""]
    
    # store path and labels
    paths, labels = [], []

    # sample images
    for g in gene_indices:
        real_paths = sample_real_images(args.real_folder, g, args.num_per_class, seed=args.seed)
        synth_paths = sample_synthetic_images(args.generated_folder, g, args.num_per_class, seed=args.seed)

        paths += real_paths + synth_paths
        labels += [f"Real-gene{g}"] * len(real_paths) + [f"Synthetic-gene{g}"] * len(synth_paths)
        
    # load model
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
Usage:
python scripts/results/Gene/umap/umap_gene_inceptionBB.py \
  --real_folder data/samples_umap/real \
  --generated_folder data/gene_36class/generated \
  --classes 0,35 \
  --num_per_class 1000 \
  --output_folder experiment/gene_36class/umaps/results_umap_inception \
  --batch_size 64 \
  --pca_components 64 \
  --n_neighbors 30 \
  --min_dist 0.10 \
  --seed 42 \
  --device cuda
"""