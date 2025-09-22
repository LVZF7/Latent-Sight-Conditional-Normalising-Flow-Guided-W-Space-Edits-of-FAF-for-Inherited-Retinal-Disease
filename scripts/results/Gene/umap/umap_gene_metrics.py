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
    genes = np.array([s.split('-', 1)[1] if '-' in s else s for s in labels])
    domains = np.array(['Real' if s.startswith('Real-') else 'Synthetic' for s in labels])
    return np.array(labels), genes, domains

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
def silhouette_gene(embedding_coords, genes):
    uniq, counts = np.unique(genes, return_counts=True)
    if len(uniq) >= 2 and np.min(counts) >= 2:
        return float(silhouette_score(embedding_coords, genes, metric='euclidean'))
    return float('nan')

# Metric 2: k-NN self-domain percentage (Real/Synthetic). Measure of how well mixed the Real and Synthetic samples are.
# formula: percent = (1/(N*k)) * sum_{i=1}^{N} count_{j in kNN(i)}(domain(j) == domain(i))
def knn_self_domain_percent(embedding_coords, domains, k=10):
    n = embedding_coords.shape[0]
    if n <= 1:
        return float('nan')
    k_excluding_self = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_excluding_self + 1, metric='euclidean').fit(embedding_coords)
    distances, neighbour_idx = nn.kneighbors(embedding_coords)
    if neighbour_idx.shape[1] > 1: # drop self (first column)
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
def centroid_distances(embedding_coords, genes, domains):
    distances = {}
    
    mask_real = (domains == 'Real')
    mask_synth = (domains == 'Synthetic')
    
    centroid_real_all = compute_centroid(embedding_coords[mask_real])
    centroid_synth_all = compute_centroid(embedding_coords[mask_synth])
    distances['centroid_dist_Real_vs_Synth_all'] = euclidean_distance(centroid_real_all, centroid_synth_all)
    
    uniq_genes = sorted(np.unique(genes).tolist())
    if len(uniq_genes) >= 2:
        gene_a, gene_b = uniq_genes[0], uniq_genes[1]
        
        # gene vs gene (0 vs 35)
        centroid_gene_a = compute_centroid(embedding_coords[genes == gene_a])
        centroid_gene_b = compute_centroid(embedding_coords[genes == gene_b])
        distances['centroid_dist_geneA_vs_geneB'] = euclidean_distance(centroid_gene_a, centroid_gene_b)
        
        # per-gene real vs synth
        centroid_gene_a_real = compute_centroid(embedding_coords[(genes == gene_a) & mask_real])
        centroid_gene_a_synth = compute_centroid(embedding_coords[(genes == gene_a) & mask_synth])
        distances['centroid_dist_geneA_Real_vs_Synth'] = euclidean_distance(centroid_gene_a_real, centroid_gene_a_synth)
        
        centroid_gene_b_real = compute_centroid(embedding_coords[(genes == gene_b) & mask_real])
        centroid_gene_b_synth = compute_centroid(embedding_coords[(genes == gene_b) & mask_synth])
        distances['centroid_dist_geneB_Real_vs_Synth'] = euclidean_distance(centroid_gene_b_real, centroid_gene_b_synth)

    return distances



# compute all the metrics for a given CSV file
def compute_metrics_for_file(csv_path, k=10):
    embedding_coords, labels, genes, domains = load_embeddings(csv_path)
    results = {
        "file": csv_path.name,
        "silhouette_gene": silhouette_gene(embedding_coords, genes),
        f"{k}NN_self_domain_pct": knn_self_domain_percent(embedding_coords, domains, k=k),
    }
    results.update(centroid_distances(embedding_coords, genes, domains))
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
python scripts/results/Gene/umap/umap_gene_metrics.py \
  --embeddings experiment/gene_36class/umaps/results_umap_genecnn/embeddings_genecnn_2d.csv \
  --k 10 \
  --output experiment/gene_36class/umaps/results_umap_genecnn/umap_metrics_genecnn_2d.csv
  
  
python scripts/results/Gene/umap/umap_gene_metrics.py \
  --embeddings experiment/gene_36class/umaps/results_umap_inception/embeddings_inception_2d.csv \
  --k 10 \
  --output experiment/gene_36class/umaps/results_umap_inception/umap_metrics_inception_2d.csv
  
  
Usage For Edited:
python scripts/results/Gene/umap/umap_gene_metrics.py \
  --embeddings experiment/gene_36class/umaps/results_umap_genecnn_FE/embeddings_genecnn_2d.csv \
  --k 10 \
  --output experiment/gene_36class/umaps/results_umap_genecnn_FE/umap_metrics_genecnn_fe_2d.csv


python scripts/results/Gene/umap/umap_gene_metrics.py \
  --embeddings experiment/gene_36class/umaps/results_umap_inception_FE/embeddings_inception_2d.csv \
  --k 10 \
  --output experiment/gene_36class/umaps/results_umap_inception_FE/umap_metrics_inception_fe_2d.csv
"""