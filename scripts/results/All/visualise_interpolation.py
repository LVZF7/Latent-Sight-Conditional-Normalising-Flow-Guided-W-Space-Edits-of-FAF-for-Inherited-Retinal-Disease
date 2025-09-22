"""
CNF Interpolation Visualisation

The script autodetects conditioning dimensions from attributes.npy (2 or 36). Loads StyleGAN2-ADA G_ema and CNF model from weights.
Does a forward pass encoding z = F(w | a_src) and then an back pass decoding w_t = F^{-1}(z | a_tgt) to get the latent code w' corresponding to the target attribute a_tgt for t in [0, 1].
saves a strip of interpolated images from a_src to a_tgt and a GIF of the interpolation.

Example usage (laterality):
python scripts/results/All/visualise_intepolation.py \
    --stylegan_weights weights/syntheye_gene_advaith_512.pkl \
    --flow_path experiment/laterality/flow/checkpoints/flow_model_final.pt \
    --latents data/laterality/numpy/latents.npy \
    --attributes data/laterality/numpy/attributes.npy \
    --steps 9 \
    --num_samples 5 \
    --start_index 0 \
    --outdir experiment/laterality/interpolations


Example usage (gene):
python scripts/results/All/visualise_intepolation.py \
    --stylegan_weights weights/syntheye_gene_advaith_512.pkl \
    --flow_path experiment/gene_36class/flow/checkpoints/flow_model_final.pt \
    --latents data/gene_36class/numpy/latents.npy \
    --attributes data/gene_36class/numpy/attributes.npy \
    --steps 9 \
    --num_samples 5 \
    --source_gene 0 \
    --target_gene 35 \
    --outdir experiment/gene_36class/interpolations

"""

import re
import os, sys, argparse
import numpy as np
import torch
from PIL import Image

# Add StyleFlow and StyleGAN2-ADA repos to path
sys.path.append('repos/StyleFlow')
sys.path.append('repos/stylegan2-ada-pytorch')

from module.flow import cnf
import dnnlib
import legacy

# Load StyleGAN2 generator from .pkl
def load_stylegan2_generator(network_pkl, device):
    if not str(network_pkl).endswith('.pkl'):
        raise ValueError("StyleGAN2 weights must be a .pkl file")
    
    print(f'Loading StyleGAN2 from {network_pkl}')
    if os.path.exists(network_pkl):
        # Local file - open directly
        with open(network_pkl, 'rb') as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)
    else:
        # URL - use dnnlib
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G

# Load CNF flow model from .pt
def load_flow_model(checkpoint_path, cond_dim, device):
    print(f'Loading CNF from {checkpoint_path}')
    model = cnf(512, '256-256-256', cond_dim, 1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.eval()

# Encode W -> Z (forward)     
def encode_wlatent(w_latent, flow_model, attr, device):
    with torch.no_grad():
        w_input = w_latent.unsqueeze(1)
        zero_pad = torch.zeros(1, 1, 1, device=device)
        z, _ = flow_model(w_input, attr, zero_pad)
        return z
    
# Decode Z -> W (reverse)
def decode_zlatent(z_latent, flow_model, attr, device):
    with torch.no_grad():
        zero_pad = torch.zeros(1, 1, 1, device=device)
        w, _ = flow_model(z_latent, attr, zero_pad, reverse=True)
        return w.squeeze(1)
    
# Generate image from w latent
def generate_image(G, w_latent, device):
    # Generate PIL image from w latent (1, 512). Broadcast W to W+ if needed
    with torch.no_grad():
        if w_latent.ndim == 2:  # (1,512)
            num_ws = getattr(getattr(G, 'mapping', None), 'num_ws', None)
            if num_ws is None:
                num_ws = getattr(getattr(G, 'synthesis', None), 'num_ws', 16)
            w_latent = w_latent.unsqueeze(1).repeat(1, num_ws, 1) # (1,num_ws,512)
        elif w_latent.ndim != 3:
            raise ValueError("w_latent must be (1,512) or (1,num_ws,512)")

        img = G.synthesis(w_latent, noise_mode='const')
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        
        c = img.shape[1]
        if c == 1:  # greyscale
            arr = img[0, 0].cpu().numpy()  # (H,W)
            return Image.fromarray(arr, mode='L')
        else:  # assume c == 3 for RGB
            arr = img[0].permute(1, 2, 0).cpu().numpy() # (H,W,3)
            return Image.fromarray(arr, mode='RGB')

# Save a strip of images horizontally
def save_image_strip(images, out_path):
    pad, bg = 6, 255
    H = min(im.height for im in images)
    resized_ims = [im.resize((int(im.width * H / im.height), H), Image.LANCZOS) for im in images] # resize to same height incase of different size outliers
    w = sum(im.width for im in resized_ims) + pad * (len(images) - 1)
    strip = Image.new('L', (w, H), bg)
    
    x = 0
    for i, im in enumerate(resized_ims):
        strip.paste(im, (x, 0))
        x += im.width + (pad if i < len(resized_ims) - 1 else 0)
    strip.save(out_path)
    print(f"Saved image strip to {out_path}")
    
# Save a GIF of images
def save_gif(images, out_path, duration_ms=200):
    # Save interpolation as GIF
    frames = [im.convert('P') for im in images] # convert to 'P' mode for GIF. greyscale 'L' images need to be converted to 'P' for GIF
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=True)
    print(f"Saved GIF to {out_path}")

# Parse specific sample ids from command line
def parse_sample_id(ids, max_n):
    if ids is None:
        return None
    out = []
    for id in ids:
        match = re.search(r'^(\d+)', str(id))
        if match:
            i = int(match.group(1))
            if 0 <= i < max_n:
                out.append(i)
        else:
            print(f"Couldn't parse sample index from '{id}'")
    return sorted(list(set(out)))
    
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stylegan_weights', required=True)
    p.add_argument('--flow_path', required=True)
    p.add_argument('--latents', required=True)
    p.add_argument('--attributes', required=True)
    p.add_argument('--steps', type=int, default=9)
    p.add_argument('--num_samples', type=int, default=5)
    p.add_argument('--start_index', type=int, default=0)
    p.add_argument('--sample_ids', nargs='+')
    p.add_argument('--source_gene', type=int, default=0)
    p.add_argument('--target_gene', type=int, default=35)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # load arrays and condition dimension
    W = np.load(args.latents) # (N,512)
    A = np.load(args.attributes) # (N, cond_dim), cond_dim is 2 for laterality, 36 for gene
    N, cond_dim = A.shape
    print(f"Loaded {N} samples with condition dimension {cond_dim}")
    
    # Load models
    G = load_stylegan2_generator(args.stylegan_weights, device)
    flow_model = load_flow_model(args.flow_path, cond_dim, device)

    specific_ids = parse_sample_id(args.sample_ids, N)
    if specific_ids is not None:
        indices = specific_ids
        print(f"Generating interpolations for specific indices {indices}")
    else:
        # choose indices
        if cond_dim == 2: # laterality
            start = args.start_index
            end = min(len(W), start + args.num_samples)
            indices = list(range(start, end))
            print(f"Generating laterality interpolations for indices {indices}")
        else: # gene
            cls = np.argmax(A, axis=1)
            cand = np.where(cls == int(args.source_gene))[0]
            if len(cand) == 0:
                print(f"No samples found with gene {args.source_gene}")
                return
            indices = cand[:args.num_samples].tolist()
            if args.target_gene >= cond_dim:
                args.target_gene = cond_dim - 1
            print(f"Generating gene interpolations from gene {args.source_gene} to {args.target_gene} for indices {indices}")
    
    
    # interpolate per sample
    ts = np.linspace(0, 1, args.steps)
    for idx in indices:
        w = torch.from_numpy(W[idx:idx+1]).float().to(device)  # (1,512)
        a = torch.from_numpy(A[idx:idx+1]).float().to(device)  # (1,cond_dim)
        
        if cond_dim == 2:
            is_left = (a[0, 0] >= a[0, 1])
            a_src = torch.tensor([[1., 0.]], device=device) if is_left else torch.tensor([[0., 1.]], device=device)
            a_tgt = torch.tensor([[0., 1.]], device=device) if is_left else torch.tensor([[1., 0.]], device=device)
            direction = "L_to_R" if is_left else "R_to_L"
        else:
            # use start and target gene indices
            a_src = torch.zeros((1, cond_dim), device=device)
            a_src[0, args.source_gene] = 1
            a_tgt = torch.zeros((1, cond_dim), device=device)
            a_tgt[0, args.target_gene] = 1
            
            direction = f"gene{args.source_gene}_to_gene{args.target_gene}"
            
        # encode to z
        z = encode_wlatent(w, flow_model, a_src, device)
        
        # decode to w_t for interpolated attributes
        images = []
        for t in ts:
            a_t = (1.0 - float(t)) * a_src + float(t) * a_tgt
            w_t = decode_zlatent(z, flow_model, a_t, device)
            img = generate_image(G, w_t, device)
            images.append(img)
            
        # save strip and gif
        filename = f"sample{idx:03d}_{direction}"
        strip_path = os.path.join(args.output_dir, filename + "_strip.png")
        gif_path = os.path.join(args.output_dir, filename + ".gif")
        save_image_strip(images, strip_path)
        save_gif(images, gif_path, duration_ms=200)
        
if __name__ == '__main__':
    main()