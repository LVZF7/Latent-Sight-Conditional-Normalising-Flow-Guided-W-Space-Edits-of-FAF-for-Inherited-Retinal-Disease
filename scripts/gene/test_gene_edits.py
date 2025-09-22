import os, sys
import numpy as np
import torch
import PIL.Image
from pathlib import Path
import argparse
import csv

# Add paths
sys.path.append('repos/StyleFlow')
sys.path.append('repos/stylegan2-ada-pytorch')

from module.flow import cnf
import dnnlib
import legacy

# load StyleGAN2 generator
def load_stylegan2_generator(network_pkl, device):
    print(f'Loading StyleGAN2 from "{network_pkl}"')
    
    # Handle both .pkl and .pt files
    if network_pkl.endswith('.pt'):
        # PyTorch checkpoint
        G = torch.load(network_pkl, map_location=device)
        if isinstance(G, dict) and 'G_ema' in G:
            G = G['G_ema']
        G = G.to(device).eval()
    else:
        # StyleGAN2-ADA pkl format - handle local files properly
        if os.path.exists(network_pkl):
            # Local file - open directly
            with open(network_pkl, 'rb') as f:
                G = legacy.load_network_pkl(f)['G_ema'].to(device)
        else:
            # URL - use dnnlib
            with dnnlib.util.open_url(network_pkl) as f:
                G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    return G

# load trained CNF model
def load_flow_model(checkpoint_path, num_classes, device):
    print(f'Loading CNF from "{checkpoint_path}"')
    model = cnf(512, '256-256-256', num_classes, 1).to(device)  # num_classes instead of hardcoded 2
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# perform gene edit
def perform_gene_edit(w_latent, flow_model, source_attr, target_attr, device):
    with torch.no_grad():
        # Prepare inputs
        w_input = w_latent.unsqueeze(1)  # [1, 1, 512]
        source_attr = source_attr.to(device)
        target_attr = target_attr.to(device)
        zero_pad = torch.zeros(1, 1, 1, device=device)
        
        # Forward: w -> z (using source attributes)
        z, _ = flow_model(w_input, source_attr, zero_pad)
        
        # Reverse: z -> w_edited (using target attributes)
        w_edited, _ = flow_model(z, target_attr, zero_pad, reverse=True)
        
        return w_edited.squeeze(1)  # [1, 512]

# generate image from W latent
def generate_image(G, w_latent, truncation_psi=0.7):
    with torch.no_grad():
        # Handle w_latent format
        if w_latent.dim() == 2:  # [1, 512]
            # Expand to w+ format
            w_latent = w_latent.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)  # [1, 18, 512]
        
        # Generate image
        img = G.synthesis(w_latent, noise_mode='const')
        print(f"Generated image shape: {img.shape}")  # Debug info
        
        # Convert to PIL Image - handle different image formats
        if img.shape[1] == 3:  # RGB
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_array = img[0].cpu().numpy()
        else:  # Grayscale or other format
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            if img.dim() == 4:  # [B, C, H, W]
                img_array = img[0].permute(1, 2, 0).cpu().numpy()
            else:
                img_array = img[0].cpu().numpy()
        
        # Ensure proper shape for PIL
        if img_array.shape[-1] == 1:  # Grayscale
            img_array = img_array.squeeze(-1)
            img_pil = PIL.Image.fromarray(img_array, 'L')
        elif img_array.shape[-1] == 3:  # RGB
            img_pil = PIL.Image.fromarray(img_array, 'RGB')
        else:
            # Handle unexpected formats
            print(f"Warning: Unexpected image shape {img_array.shape}, converting to grayscale")
            if len(img_array.shape) == 3:
                img_array = img_array.mean(axis=-1)
            img_pil = PIL.Image.fromarray(img_array.astype(np.uint8), 'L')
        
        return img_pil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stylegan_pkl', default='weights/syntheye_gene_advaith_512.pkl')
    parser.add_argument('--flow_checkpoint', default='experiment/gene/flow/checkpoints/flow_model_final.pt')
    parser.add_argument('--latents_path', default='data/gene/numpy/latents.npy')
    parser.add_argument('--attrs_path', default='data/gene/numpy/attributes.npy')
    parser.add_argument('--output_dir', default='experiment/gene/edits')
    parser.add_argument('--source_gene', type=int, required=True)
    parser.add_argument('--target_gene', type=int, required=True)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    latents_dir = os.path.join(args.output_dir, 'latents')
    os.makedirs(latents_dir, exist_ok=True)

    # Load data to determine number of classes
    attrs = np.load(args.attrs_path)
    num_classes = attrs.shape[1]  # Auto-detect from one-hot encoding
    print(f"Detected {num_classes} gene classes")

    # Load models
    G = load_stylegan2_generator(args.stylegan_pkl, device)
    flow_model = load_flow_model(args.flow_checkpoint, num_classes, device)

    # Load test data
    latents = np.load(args.latents_path)  # [N, 512]
    attrs = np.load(args.attrs_path)      # [N, num_classes]
    
    print(f"Loaded data:")
    print(f"- Latents shape: {latents.shape}")
    print(f"- Attributes shape: {attrs.shape}")

    # Find samples from source gene class ONLY
    source_mask = attrs[:, args.source_gene] == 1
    source_indices = np.where(source_mask)[0]
    
    print(f"Found {len(source_indices)} samples from source gene class {args.source_gene}")
    
    if len(source_indices) == 0:
        raise ValueError(f"No samples found for source gene class {args.source_gene}")
    
    # Select samples to edit
    if args.num_samples > 0 and len(source_indices) > args.num_samples:
        selected_indices = np.random.choice(source_indices, args.num_samples, replace=False)
    else:
        selected_indices = source_indices[:args.num_samples] if args.num_samples > 0 else source_indices
    
    print(f"Using {len(selected_indices)} samples for editing")

    # Create manifest file
    man_path = os.path.join(args.output_dir, "edits_manifest.csv")
    is_new = not os.path.exists(man_path)
    with open(man_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "sample_id","orig_path","edited_path","reversed_path",
                "source_side","target_side","edit_magnitude","reverse_error"
            ])

    # Process each sample
    for i, idx in enumerate(selected_indices):
        print(f"\nProcessing sample {i+1}/{len(selected_indices)} (Index: {idx})")
        
        # Get original latent
        w_orig = torch.from_numpy(latents[idx:idx+1]).to(device)  # [1, 512]
        
        # Generate original image
        img_orig = generate_image(G, w_orig)
        
        # Define source and target attributes  
        source_attr = torch.zeros(1, num_classes, device=device)
        source_attr[0, args.source_gene] = 1.0
        target_attr = torch.zeros(1, num_classes, device=device)
        target_attr[0, args.target_gene] = 1.0
        
        # Use gene class names for consistency
        source_side = f"gene{args.source_gene}"
        target_side = f"gene{args.target_gene}"

        print(f"Original: {source_side} -> Editing to: {target_side}")

        # Perform gene edit
        w_edited = perform_gene_edit(w_orig, flow_model, source_attr, target_attr, device)
        
        # Generate edited image
        img_edited = generate_image(G, w_edited)
        
        # Test reverse edit
        w_reverse = perform_gene_edit(w_edited, flow_model, target_attr, source_attr, device)
        img_reverse = generate_image(G, w_reverse)
        
        # Calculate edit magnitude
        edit_magnitude = torch.norm(w_edited - w_orig).item()
        reverse_error = torch.norm(w_reverse - w_orig).item()
        
        # Save images using consistent naming
        base = f"sample_{i:03d}"
        orig_path = os.path.join(args.output_dir, f"{base}_original.png")
        edit_path = os.path.join(args.output_dir, f"{base}_edited.png")
        rev_path  = os.path.join(args.output_dir, f"{base}_reversed.png")

        img_orig.save(orig_path)
        img_edited.save(edit_path)
        img_reverse.save(rev_path)
        
        # Save latents for traceability
        np.savez(os.path.join(latents_dir, f"{base}_original_w.npz"), w=w_orig.detach().cpu().numpy())
        np.savez(os.path.join(latents_dir, f"{base}_edited_w.npz"),  w=w_edited.detach().cpu().numpy())
        np.savez(os.path.join(latents_dir, f"{base}_reversed_w.npz"),w=w_reverse.detach().cpu().numpy())
        print("Saved latents for original, edited, reversed")

        # Append row to manifest
        with open(man_path, "a", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([
                base, orig_path, edit_path, rev_path,
                source_side, target_side, f"{edit_magnitude:.6f}", f"{reverse_error:.6f}"
            ])

        print(f"Edit magnitude: {edit_magnitude:.4f}")
        print(f"Reverse error: {reverse_error:.4f}")
        print(f"Saved: original, edited, reversed + latents")

    print(f"\nAll edits completed! Results saved to: {args.output_dir}")
    print("Files generated:")
    print("- *_original.png: Original images")
    print("- *_edited.png: Edited images")
    print("- *_reversed.png: Reverse edits (should match original)")
    print("- *_w.npz: Latent codes for traceability")
    print("- edits_manifest.csv: Metadata and metrics")

if __name__ == '__main__':
    main()
