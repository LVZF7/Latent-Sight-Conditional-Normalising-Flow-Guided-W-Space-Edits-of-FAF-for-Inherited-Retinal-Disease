import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from sklearn.linear_model import LogisticRegression

import lpips
from skimage.metrics import structural_similarity as ssim



# load laterality cnn model
def load_cnn_model(checkpoint_path, device):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2) 
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]  # Extract the actual model state dict
    state_dict = {key.replace("module.", ""): val for key, val in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    
    model.fc = nn.Identity()  # remove final layer to get features
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # preprocessing transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
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
    
    
# eXtract features using the model
@torch.no_grad()
def extract_features(img_paths, model, preprocess, device, batch_size=64):
    dataset = ImageDataset(img_paths, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    features = []
    for imgs, paths in dataloader:
        imgs = imgs.to(device)
        feats = model(imgs).cpu().numpy().astype(np.float32)
        features.append(feats)
    return np.concatenate(features, axis=0)


# convert to lpips tensor
def lpips_tensor(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    lpips_tensor = transform(img) * 2 - 1  # scale to [-1, 1]
    return lpips_tensor.unsqueeze(0)  # add batch dimension

# average lpips distance
@torch.no_grad()
def avg_lpips(pairs, device, lpips_net):
    val = []
    for img1, img2 in pairs:
        img1_tensor = lpips_tensor(Image.open(img1).convert('RGB')).to(device)
        img2_tensor = lpips_tensor(Image.open(img2).convert('RGB')).to(device)
        dist = lpips_net(img1_tensor, img2_tensor).item()
        val.append(dist)
    return np.mean(val) if val else np.nan

# average ssim
def avg_ssim(pairs):
    val = []
    for img1, img2 in pairs:
        im1 = np.array(Image.open(img1).convert('L').resize((224, 224)))
        im2 = np.array(Image.open(img2).convert('L').resize((224, 224)))
        s = ssim(im1, im2, data_range=255)
        val.append(s)
    return np.mean(val) if val else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--triplet_csv', type=Path, required=True)
    p.add_argument('--cnn_checkpoint', type=Path, required=True)
    p.add_argument('--output_csv', type=Path, required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    df = pd.read_csv(args.triplet_csv)
    orig_paths = df['orig_path'].astype(str).tolist()
    edit_paths = df['edited_path'].astype(str).tolist()
    rev_paths = df['reversed_path'].astype(str).tolist()
    num_paths = len(orig_paths)

    # load CNN model
    model, preprocess = load_cnn_model(args.cnn_checkpoint, device)

    # extract features
    feat_orig = extract_features(orig_paths, model, preprocess, device, args.batch_size)
    feat_edit = extract_features(edit_paths, model, preprocess, device, args.batch_size)
    feat_rev = extract_features(rev_paths, model, preprocess, device, args.batch_size)

    # l2 normalisation
    feat_orig = feat_orig / np.linalg.norm(feat_orig, axis=1, keepdims=True)
    feat_edit = feat_edit / np.linalg.norm(feat_edit, axis=1, keepdims=True)
    feat_rev = feat_rev / np.linalg.norm(feat_rev, axis=1, keepdims=True)
    
    
    x = np.vstack([feat_orig, feat_edit])
    y = np.concatenate([np.zeros(num_paths), np.ones(num_paths)])
    clf = LogisticRegression(random_state=args.seed, max_iter=1000).fit(x, y)
    
    pred_orig = clf.predict(feat_orig)
    pred_edit = clf.predict(feat_edit)
    pred_rev = clf.predict(feat_rev)
    
    # crossing and reversal metrics
    crossing_rate = np.mean(pred_orig != pred_edit) * 100
    reversal_rate = np.mean(pred_orig == pred_rev) * 100
    
    # lpips and ssim metrics
    lpips_net = lpips.LPIPS(net='alex').to(device).eval()
    
    edit_pairs = list(zip(orig_paths, edit_paths))
    rev_pairs = list(zip(orig_paths, rev_paths))
    lpips_edit = avg_lpips(edit_pairs, device, lpips_net)
    lpips_rev = avg_lpips(rev_pairs, device, lpips_net)
    ssim_edit = avg_ssim(edit_pairs)
    ssim_rev = avg_ssim(rev_pairs)
    
    # save results
    results = {
        'Crossing %': crossing_rate,
        'Reversal %': reversal_rate,
        'LPIPS(Original, Edited)': lpips_edit,
        'LPIPS(Original, Reversed)': lpips_rev,
        'SSIM(Original, Edited)': ssim_edit,
        'SSIM(Original, Reversed)': ssim_rev,
    }
    results_df = pd.DataFrame([results])
    
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    
if __name__ == '__main__':
    main()
    
"""
Usage (Laterality Perception Metrics):
python scripts/results/All/perception_metrics.py \
    --triplet_csv experiment/laterality/edits/edits_manifest.csv \
    --cnn_checkpoint weights/laterality_120epoch/best_model.pt \
    --output_csv experiment/laterality/metrics/lat_eval_metrics.csv \
    --device cuda \
    --batch_size 64 \
    --seed 42
    
    
Usage (Gene Perception Metrics):
python scripts/results/All/perception_metrics.py \
    --triplet_csv experiment/gene_36class/edits/edits_manifest.csv \
    --cnn_checkpoint weights/gene_120epoch/best_model.pt \
    --output_csv experiment/gene_36class/metrics/gene_eval_metrics.csv \
    --device cuda \
    --batch_size 64 \
    --seed 42
"""