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

# Load trained CNF model
def load_flow_model(checkpoint_path, device):
    print(f'Loading CNF from "{checkpoint_path}"')
    model = cnf(512, '256-256-256', 2, 1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# Perform laterality edit using CNF
def perform_laterality_edit(w_latent, flow_model, source_attr, target_attr, device):
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

# Generate image from W latent using StyleGAN2
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
            print(f"Warning - Unexpected image shape {img_array.shape}, converting to grayscale")
            if len(img_array.shape) == 3:
                img_array = img_array.mean(axis=-1)
            img_pil = PIL.Image.fromarray(img_array.astype(np.uint8), 'L')
        
        return img_pil

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stylegan_pkl', default='weights/syntheye_gene_advaith_512.pt')
    p.add_argument('--flow_checkpoint', default='experiments/flow/checkpoints/laterality_model_final.pt')
    p.add_argument('--latents_path', default='data_numpy/latents.npy')
    p.add_argument('--attrs_path', default='data_numpy/attributes.npy')
    p.add_argument('--output_dir', default='experiments/edits')
    p.add_argument('--num_samples', type=int, default=5)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    latents_dir = os.path.join(args.output_dir, 'latents')
    os.makedirs(latents_dir, exist_ok=True)

    # Load models
    G = load_stylegan2_generator(args.stylegan_pkl, device)
    flow_model = load_flow_model(args.flow_checkpoint, device)

    # Load test data
    latents = np.load(args.latents_path)[:args.num_samples]  # [N, 512]
    attrs = np.load(args.attrs_path)[:args.num_samples]      # [N, 2]
    
    print(f"Loaded {len(latents)} test samples")
    print(f"Latents shape: {latents.shape}")
    print(f"Attributes shape: {attrs.shape}")

    # Define laterality attributes
    left_attr = torch.tensor([[1.0, 0.0]], device=device)   # Left side
    right_attr = torch.tensor([[0.0, 1.0]], device=device)  # Right side
    
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
    for i in range(len(latents)):
        print(f"\nProcessing sample {i+1}/{len(latents)}")
        
        # Get original latent and attribute
        w_orig = torch.from_numpy(latents[i:i+1]).to(device)  # [1, 512]
        orig_attr = torch.from_numpy(attrs[i:i+1]).to(device)  # [1, 2]
        
        # Generate original image
        img_orig = generate_image(G, w_orig)
        ## img_orig.save(os.path.join(args.output_dir, f'sample_{i:03d}_original.png'))
        
        # Determine source laterality
        if orig_attr[0, 0] > orig_attr[0, 1]:
            source_side = "left"
            source_attr = left_attr
            target_attr = right_attr
            target_side = "right"
        else:
            source_side = "right"
            source_attr = right_attr
            target_attr = left_attr
            target_side = "left"
        
        print(f"  Original: {source_side} -> Editing to: {target_side}")
        
        # Perform laterality edit
        w_edited = perform_laterality_edit(w_orig, flow_model, source_attr, target_attr, device)
        
        # Generate edited image
        img_edited = generate_image(G, w_edited)
        ## img_edited.save(os.path.join(args.output_dir, f'sample_{i:03d}_{source_side}_to_{target_side}.png'))
        
        # Also test the reverse edit
        w_reverse = perform_laterality_edit(w_edited, flow_model, target_attr, source_attr, device)
        img_reverse = generate_image(G, w_reverse)
        ## img_reverse.save(os.path.join(args.output_dir, f'sample_{i:03d}_{target_side}_to_{source_side}_reverse.png'))
        
        # Calculate edit magnitude
        edit_magnitude = torch.norm(w_edited - w_orig).item()
        reverse_error = torch.norm(w_reverse - w_orig).item()
        
        # Save images
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

        print(f"  Edit magnitude: {edit_magnitude:.4f}")
        print(f"  Reverse error: {reverse_error:.4f}")
        print(f"  Saved: original, edited, reversed + latents")

    print(f"\nAll edits completed! Results saved to: {args.output_dir}")
    print("Files generated:")
    print("- *_original.png: Original images")
    print("- *_edited.png: Edited images")
    print("- *_reversed.png: Reverse edits (should match original)")
    print("- *_w.npz: Latent codes for traceability")
    print("- edits_manifest.csv: Metadata and metrics")

if __name__ == '__main__':
    main()