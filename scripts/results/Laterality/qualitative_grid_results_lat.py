# Goal: Create a grid visualisation of editing results with headers: Original Synthetic | Edited Synthetic | Reversed Edited Synthetic

import os
import glob
import argparse
import csv
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

# Helper to center text at (x_center, y)
def _text_center(draw, text, x_center, y, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = right - left
    draw.text((x_center - width // 2, y), text, fill='black', font=font)

# Load edits_manifest.csv mapping sample number -> (source_side, target_side)
def load_manifest(manifest_path):
    mapping = {}
    if not manifest_path or not os.path.exists(manifest_path):
        return mapping
    with open(manifest_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get('sample_id') or row.get('sample') or ''
            if not sample_id.startswith('sample_'):
                continue
            # sample_000 -> 000
            sample_num = sample_id.split('_')[1]
            source = (row.get('source_side') or '').strip().lower()
            target = (row.get('target_side') or '').strip().lower()
            if source and target:
                mapping[sample_num] = {
                    'source': source,
                    'target': target
                }
    return mapping

# Create grid of results
def create_grid(input_dir, output_dir, img_size=128, manifest_path=None, num_samples=None):
    manifest_map = load_manifest(manifest_path)
    if manifest_path:
        if manifest_map:
            print(f"Loaded manifest for {len(manifest_map)} samples: {manifest_path}")
        else:
            print(f"Manifest not found or empty: {manifest_path}")

    # Find synthetic originals
    original_files = sorted(glob.glob(os.path.join(input_dir, 'sample_*_original.png')))
    total_samples = len(original_files)
    
    # Limit number of samples if needed
    if num_samples is not None and num_samples > 0:
        original_files = original_files[:num_samples]
        used_samples = len(original_files)
        print(f"Using {used_samples} out of {total_samples} available samples")
    else:
        used_samples = total_samples
        print(f"Using all {used_samples} samples")
    
    if used_samples == 0:
        print("No samples found")
        return

    # font, size and layout
    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True) # using matplotlibs font manager so that it works across different OS's.
    font = ImageFont.truetype(font_path, 14)
    asc, desc = font.getmetrics()
    line_h = asc + desc + 2
    header_h = line_h + 6
    label_h = line_h 

    # Layout constants
    cols = 3
    pad = 16
    total_w = cols * img_size + (cols + 1) * pad
    total_h = header_h + pad + used_samples * (img_size + label_h + pad)

    # Canvas and font
    grid = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(grid)

    # Column headers
    headers = ['Original Synthetic', 'Edited Synthetic', 'Reversed Edited Synthetic']
    for col, header in enumerate(headers):
        x_center = pad + col * (img_size + pad) + img_size // 2
        _text_center(draw, header, x_center, 2, font)

    # Rows
    for row, orig_path in enumerate(original_files):
        basename = os.path.basename(orig_path)
        parts = basename.split('_')
        if len(parts) < 3:
            continue
        sample_num = parts[1]

        edited_path = os.path.join(input_dir, f'sample_{sample_num}_edited.png')
        reverse_path = os.path.join(input_dir, f'sample_{sample_num}_reversed.png')

        if not os.path.exists(edited_path) or not os.path.exists(reverse_path):
            print(f"Missing files for sample {sample_num}")
            continue

        try:
            orig_img = Image.open(orig_path).convert('RGB').resize((img_size, img_size), Image.LANCZOS)
            edited_img = Image.open(edited_path).convert('RGB').resize((img_size, img_size), Image.LANCZOS)
            rev_img = Image.open(reverse_path).convert('RGB').resize((img_size, img_size), Image.LANCZOS)
        except Exception as e:
            print(f"Error loading images for sample {sample_num}: {e}")
            continue

        images = [orig_img, edited_img, rev_img]

        # Determine labels using manifest
        if sample_num in manifest_map:
            source = manifest_map[sample_num]['source']
            target = manifest_map[sample_num]['target']
            # Laterality abbreviations (e.g. Left -> L, Right -> R)
            s_abbr = source[0].upper()
            t_abbr = target[0].upper()

            # Original labels
            orig_labels = [f"Sample {sample_num} {s_abbr}"]

            # Edited: source to target
            edit_labels = [f"{s_abbr} to {t_abbr}"]

            # Reverse: target to source
            rev_labels = [f"{t_abbr} to {s_abbr}"]

            label_lines = [orig_labels, edit_labels, rev_labels]
        else:
            # backup if no manifest
            label_lines = [
                [f"Sample {sample_num}"],
                ["Source to Target"],
                ["Target to Source"]
            ]

        # Paste images and labels into grid
        top_y = header_h + pad + row * (img_size + label_h + pad) # Y position of top of this row
        for col, (img, lines) in enumerate(zip(images, label_lines)):
            left_x = pad + col * (img_size + pad) # X position of left of this column
            x_center = left_x + img_size // 2 # Center X of this column
            grid.paste(img, (left_x, top_y)) # Paste image
            cur_y = top_y + img_size + 4 # Y position for labels
            for line in lines: 
                _text_center(draw, line, x_center, cur_y, font) # Draw each line of label
                cur_y += line_h # Move down for next line

    filename = f"lat_grid_{used_samples}_samples"
    png_path = os.path.join(output_dir, f"{filename}.png")
    
    os.makedirs(output_dir, exist_ok=True)
    grid.save(png_path, quality=95) # Save as PNG
    print(f"PNG saved: {png_path}")

    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    grid.save(pdf_path, format='PDF') # Save as PDF
    print(f"PDF saved: {pdf_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', default='experiment/laterality/edits')
    p.add_argument('--output_dir', default='experiment/laterality/result_grid')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--manifest', default='experiment/laterality/edits/edits_manifest.csv')
    p.add_argument('--num_samples', type=int, default=None,)
    args = p.parse_args()

    create_grid(
        args.input_dir,
        args.output_dir,
        img_size=args.img_size,
        manifest_path=args.manifest,
        num_samples=args.num_samples
    )

if __name__ == '__main__':
    main()