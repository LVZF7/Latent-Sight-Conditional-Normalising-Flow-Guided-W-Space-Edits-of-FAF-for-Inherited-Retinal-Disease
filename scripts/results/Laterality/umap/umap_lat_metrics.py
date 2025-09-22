import argparse
from pathlib import Path
import glob
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# Split Real-L/Synthetic-R into side {'L','R'} and domain {'Real','Synthetic'}.
def parse_labels(label_series):
    labels = label_series.astype(str).tolist()
    sides = np.array(['L' if '-L' in s else 'R' for s in labels])
    domains = np.array(['Real' if s.startswith('Real-') else 'Synthetic' for s in labels])
    return np.array(labels), sides, domains

# load embeddings from CSV file, return X (N×2 or N×3), labels, sides, domains
def load_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    coordinate_cols = [c for c in ['x', 'y', 'z'] if c in df.columns]
    if len(coordinate_cols) < 2:
        raise ValueError(f"{csv_path}: embeddings CSV must include at least ['x','y'] columns.")
    X = df[coordinate_cols].to_numpy(dtype=np.float64)
    labels, sides, domains = parse_labels(df['label'])
    return X, labels, sides, domains


# Metrics:

# Metric 1: Silhouette for laterality clusters (L vs R). Measure of how well-separated the L and R clusters are.
# formula: s = (b - a) / max(a, b)
def silhouette_laterality(X, sides):
    uniq, counts = np.unique(sides, return_counts=True)
    if len(uniq) >= 2 and np.min(counts) >= 2:
        return float(silhouette_score(X, sides, metric='euclidean'))
    return float('nan')

# Metric 2: k-NN self-domain percentage (Real/Synthetic). Measure of how well mixed the Real and Synthetic samples are.
# formula: percent = (1/(N*k)) * sum_{i=1}^{N} count_{j in kNN(i)}(domain(j) == domain(i))
def knn_self_domain_percent(X, domains, k=10):
    n = X.shape[0]
    if n <= 1:
        return float('nan')
    k_excluding_self = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_excluding_self + 1, metric='euclidean').fit(X)
    distances, neighbour_idx = nn.kneighbors(X)
    # Exclude self (first column)
    if neighbour_idx.shape[1] > 1:
        neighbour_idx = neighbour_idx[:, 1:]
    same_domain_frac = (domains[neighbour_idx] == domains[:, None]).mean(axis=1)
    percent = same_domain_frac.mean() * 100.0
    return float(percent)


# Metric 3: Centroid distances between various groups. Helps understand spatial relationships.
# 3.1 helper function to compute centroid of a set of points
def compute_centroid(points):
    if points.shape[0] == 0:
        raise ValueError("No points to compute centroid.")
    return np.mean(points, axis=0)

# 3.2 helper function to compute Euclidean distance between two points
def euclidean_distance(point_vec1, point_vec2):
    if point_vec1.shape != point_vec2.shape:
        raise ValueError("Point vectors must have the same dimensions.")
    return np.linalg.norm(point_vec1 - point_vec2)


# M3: compute centroid distances between various groups
def centroid_distances(embedding_coords, sides, domains):
    # boolean masks for each group
    mask_L = (sides == 'L')
    mask_R = (sides == 'R')
    mask_Real = (domains == 'Real')
    mask_Synthetic = (domains == 'Synthetic')
    mask_Real_L = mask_L & mask_Real
    mask_Real_R = mask_R & mask_Real
    mask_Synthetic_L = mask_L & mask_Synthetic
    mask_Synthetic_R = mask_R & mask_Synthetic
    
    # compute centroids
    centroid_L = compute_centroid(embedding_coords[mask_L])
    centroid_R = compute_centroid(embedding_coords[mask_R])
    centroid_Real = compute_centroid(embedding_coords[mask_Real])
    centroid_Synthetic = compute_centroid(embedding_coords[mask_Synthetic])
    centroid_Real_L = compute_centroid(embedding_coords[mask_Real_L])
    centroid_Real_R = compute_centroid(embedding_coords[mask_Real_R])
    centroid_Synthetic_L = compute_centroid(embedding_coords[mask_Synthetic_L])
    centroid_Synthetic_R = compute_centroid(embedding_coords[mask_Synthetic_R])
    
    # compute distances between centroids
    distances = {
        "centroid_dist_L_vs_R": euclidean_distance(centroid_L, centroid_R),
        "centroid_dist_RealL_vs_SynthL": euclidean_distance(centroid_Real_L, centroid_Synthetic_L),
        "centroid_dist_RealR_vs_SynthR": euclidean_distance(centroid_Real_R, centroid_Synthetic_R),
        "centroid_dist_Real_vs_Synth_all": euclidean_distance(centroid_Real, centroid_Synthetic),
    }   
    
    return distances



# compute all the metrics for a given CSV file
def compute_metrics_for_file(csv_path, k=10):
    X, labels, sides, domains = load_embeddings(csv_path)
    results = {
        "file": csv_path.name,
        "silhouette_LR": silhouette_laterality(X, sides),
        f"{k}NN_self_domain_pct": knn_self_domain_percent(X, domains, k=k),
    }
    results.update(centroid_distances(X, sides, domains))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=str, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    
    # Expand glob pattern
    files = sorted([Path(f) for f in glob.glob(args.embeddings)])
    if not files:
        raise FileNotFoundError(f"No files matched the pattern: {args.embeddings}")
    
    # Compute metrics for each file
    rows = []
    for f in files:
        try:
            rows.append(compute_metrics_for_file(f, k=args.k))
        except Exception as e:
            rows.append({"file": f.name, "error": str(e)})
            print(f"Error processing {f}: {e}")
            
    # Save results to CSV
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved metrics to: {args.output.resolve()}")
        
if __name__ == "__main__":
    main()
    
"""
Usage:
python scripts/results/Laterality/umap/umap_lat_metrics.py \
  --embeddings experiment/laterality/umaps/results_umap_latcnn/embeddings_latcnn_2d.csv \
  --k 10 \
  --output experiment/laterality/umaps/results_umap_latcnn/umap_metrics_latcnn_2d.csv
  
  
python scripts/results/Laterality/umap/umap_lat_metrics.py \
  --embeddings experiment/laterality/umaps/results_umap_inception/embeddings_inception_2d.csv \
  --k 10 \
  --output experiment/laterality/umaps/results_umap_inception/umap_metrics_inception_2d.csv



Usage For Edited:

python scripts/results/Laterality/umap/umap_lat_metrics.py \
  --embeddings experiment/laterality/umaps/results_umap_latcnn_FE/embeddings_latcnn_2d.csv \
  --k 10 \
  --output experiment/laterality/umaps/results_umap_latcnn_FE/umap_metrics_latcnn_fe_2d.csv

python scripts/results/Laterality/umap/umap_lat_metrics.py \
  --embeddings experiment/laterality/umaps/results_umap_inception_FE/embeddings_inception_2d.csv \
  --k 10 \
  --output experiment/laterality/umaps/results_umap_inception_FE/umap_metrics_inception_fe_2d.csv
"""