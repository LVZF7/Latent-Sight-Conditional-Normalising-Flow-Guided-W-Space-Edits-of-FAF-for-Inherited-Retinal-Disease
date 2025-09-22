import pandas as pd
import os
import shutil
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Load the laterality classifier model
def load_laterality_model(model_path, device):
    # Load the trained laterality classifier.
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)  # 2 classes: L and R
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model

# Image transformations
def get_transform():
    # Get image transform for laterality classifier.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

# Predict laterality with confidence score
def predict_laterality(model, image_path, transform, device):
    # Predict laterality with confidence score.
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            predicted_label = "L" if predicted_class == 0 else "R"
            
        return predicted_label, confidence
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='data/real/nnunet_faf_v0_dataset_v2_local.csv')
    p.add_argument('--model_path', default='weights/laterality_20epoch_full/best_model.pt')
    p.add_argument('--output_dir', default='data/samples_umap/real')
    p.add_argument('--n_samples', type=int, default=1000, help='Number of samples per side')
    p.add_argument('--confidence_threshold', type=float, default=0.99)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load laterality model
    print("Loading laterality classifier")
    model = load_laterality_model(args.model_path, device)
    transform = get_transform()
    
    # Load CSV
    df = pd.read_csv(args.csv)
    total_images = len(df)
    print(f"Processing {total_images} images")
    
    # Create output directories
    out_L = Path(args.output_dir) / 'L'
    out_R = Path(args.output_dir) / 'R'
    out_L.mkdir(parents=True, exist_ok=True)
    out_R.mkdir(parents=True, exist_ok=True)
    
    # Counters
    saved_L = 0
    saved_R = 0
    processed = 0
    
    # Process all images
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if saved_L >= args.n_samples and saved_R >= args.n_samples:
            break
            
        src = row['file_path_original']
        if not os.path.exists(src):
            continue
            
        # Predict laterality
        pred_label, confidence = predict_laterality(model, src, transform, device)
        processed += 1
        
        # Save if confidence threshold met and we need more of this side
        if confidence >= args.confidence_threshold:
            if pred_label == 'L' and saved_L < args.n_samples:
                dst = out_L / f"L_{saved_L:04d}_{Path(src).name}"
                shutil.copy2(src, dst)
                saved_L += 1
            elif pred_label == 'R' and saved_R < args.n_samples:
                dst = out_R / f"R_{saved_R:04d}_{Path(src).name}"
                shutil.copy2(src, dst)
                saved_R += 1
    
    print(f"Processed {processed} images out of {total_images} total")
    print(f"Saved {saved_L} L images to {out_L}")
    print(f"Saved {saved_R} R images to {out_R}")
    print(f"Confidence threshold: {args.confidence_threshold}")

if __name__ == '__main__':
    main()