import argparse
import subprocess
import os
from pathlib import Path

# Generate data for a specific gene class
def generate_gene_class_data(network_pkl, gene_class, num_samples, base_seed, output_dir, trunc=0.7):
    print(f"Generating {num_samples} samples for gene class {gene_class}")
    
    # Calculate seed range for this class
    start_seed = base_seed + (gene_class * num_samples)
    end_seed = start_seed + num_samples - 1
    
    # Create class-specific output directory
    class_output_dir = Path(output_dir) / f"gene_{gene_class}"
    
    # Run generation command
    cmd = [
        "python", "repos/stylegan2-ada-pytorch/generateGSCNFMod.py",
        "--network", network_pkl,
        "--seeds", f"{start_seed}-{end_seed}",
        "--trunc", str(trunc),
        "--class", str(gene_class),
        "--outdir", str(class_output_dir),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    print(f"Generated gene class {gene_class} data in {class_output_dir}")
    return class_output_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='weights/syntheye_gene_advaith_512.pkl')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--samples_per_class', type=int, default=200)
    parser.add_argument('--base_seed', type=int, default=1000)
    parser.add_argument('--trunc', type=float, default=0.7)
    parser.add_argument('--output_dir', default='data/gene/generated')
    args = parser.parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Starting gene synthetic data generation:")
    print(f"- Network: {args.network}")
    print(f"- Classes: {args.num_classes}")
    print(f"- Samples per class: {args.samples_per_class}")
    print(f"- Output: {args.output_dir}")
    print()
    
    # Generate data for each gene class
    class_dirs = []
    for gene_class in range(args.num_classes):
        class_dir = generate_gene_class_data(
            network_pkl=args.network,
            gene_class=gene_class,
            num_samples=args.samples_per_class,
            base_seed=args.base_seed,
            output_dir=args.output_dir,
            trunc=args.trunc
        )
        class_dirs.append(class_dir)
    
    print("\n" + "-"*50)
    print("Gene synthetic data generation complete")
    print(f"Generated {args.num_classes} classes x {args.samples_per_class} samples")
    print(f"Total samples: {args.num_classes * args.samples_per_class}")
    print("\nGenerated directories:")
    for class_dir in class_dirs:
        print(f"- {class_dir}")

if __name__ == '__main__':
    main()