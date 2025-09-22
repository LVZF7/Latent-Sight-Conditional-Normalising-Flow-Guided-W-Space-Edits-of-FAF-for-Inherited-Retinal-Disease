import os, argparse, csv
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# load laterality model
def load_laterality_model(ckpt_path, device):
    import torchvision.models as tvm
    
    # Use pretrained=False instead of weights=None for older PyTorch
    model = tvm.resnet18(pretrained=False, num_classes=2)
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract state_dict from checkpoint structure
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

# make image transformations
def make_transform(min_size):
    # Use ImageNet normalization (same as your training script)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize(min_size),
        transforms.CenterCrop(min_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--ckpt',     required=True)
    p.add_argument('--min_size', type=int, default=224)
    p.add_argument('--batch',    type=int, default=32)
    p.add_argument('--col_pred', default='lat_pred')
    p.add_argument('--col_conf', default='lat_conf')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_laterality_model(args.ckpt, device)
    tfm = make_transform(args.min_size)

    df = pd.read_csv(args.manifest)
    preds, confs = [], []

    print(f"Processing {len(df)} images...")
    
    # Use torch.no_grad() instead of torch.inference_mode()
    with torch.no_grad():
        for idx, row in df.iterrows():
            img_path = row['img_path']
            
            # Load and convert image to RGB
            img = Image.open(img_path).convert('RGB')
            x = tfm(img).unsqueeze(0).to(device)
            
            # Get predictions
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].detach().cpu()
            pred_idx = int(torch.argmax(probs).item())
            pred_label = 'L' if pred_idx == 0 else 'R'
            conf = float(probs[pred_idx].item())
            
            preds.append(pred_label)
            confs.append(conf)
            
            if (idx+1) % 50 == 0:
                print(f"[{idx+1}/{len(df)}] {img_path} -> {pred_label} ({conf:.3f})")

    # Add predictions to dataframe and save
    df[args.col_pred] = preds
    df[args.col_conf] = confs
    df.to_csv(args.manifest, index=False)
    
    # Print summary
    l_count = sum(1 for p in preds if p == 'L')
    r_count = sum(1 for p in preds if p == 'R')
    avg_conf = sum(confs) / len(confs)
    
    print(f"\nResults summary:")
    print(f"- Left eye predictions: {l_count}")
    print(f"- Right eye predictions: {r_count}")
    print(f"- Average confidence: {avg_conf:.3f}")
    print(f"Manifest updated: {args.manifest}")

if __name__ == '__main__':
    main()