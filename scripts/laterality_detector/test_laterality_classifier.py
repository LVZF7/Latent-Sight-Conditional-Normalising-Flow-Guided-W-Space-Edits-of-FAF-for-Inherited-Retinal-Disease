"""
Test the trained laterality classifier on real images and visualize predictions.

Usage:
    python test_laterality_classifier.py \
        --model-path checkpoints/laterality_20epoch_full/best_model.pt \
        --image-dir data/real_image_splits/test \
        --output-dir test_predictions \
        --num-samples 16

"""

import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import json

# dedicated parsing function
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default="test_predictions")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--backbone", type=str, default="resnet18")
    return p.parse_args()

# Function to load the model
def load_model(model_path, backbone, device):
    # Create model architecture
    if backbone.lower() == "resnet18":
        model = models.resnet18(pretrained=False)  # Use pretrained instead of weights
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)  # 2 classes: L and R
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Model performance from checkpoint: {checkpoint.get('metrics', 'N/A')}")
    
    return model

# Image preprocessing function
def get_test_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

# Function to collect test images
def collect_test_images(image_dir, num_samples):
    left_dir = image_dir / "L"
    right_dir = image_dir / "R" 
    
    # Look for images in nested directory structure: .pat/.sdb/.png
    left_images = []
    right_images = []
    
    def find_images_in_nested_dirs(base_dir):
        images = []
        for pat_dir in base_dir.iterdir():
            if pat_dir.is_dir() and pat_dir.suffix == '.pat':
                for sdb_dir in pat_dir.iterdir():
                    if sdb_dir.is_dir() and sdb_dir.suffix == '.sdb':
                        # Look for PNG files in the .sdb directory
                        png_files = list(sdb_dir.glob("*.png"))
                        images.extend(png_files)
        return images
    
    # Search for images in left directory
    left_images = find_images_in_nested_dirs(left_dir)
    
    # Search for images in right directory
    right_images = find_images_in_nested_dirs(right_dir)
    
    print(f"Found {len(left_images)} left eye images and {len(right_images)} right eye images")
    
    if len(left_images) == 0 and len(right_images) == 0:
        print(f"No images found in {image_dir}")
        
        # Debug: Show directory structure
        print("Directory structure sample:")
        for i, item in enumerate(left_dir.rglob("*")):
            if i >= 10:  # Limit output
                break
            print(f"  {item}")
        
        return []
    
    # Sample equal numbers from each class
    samples_per_class = num_samples // 2
    
    left_sample = random.sample(left_images, min(samples_per_class, len(left_images)))
    right_sample = random.sample(right_images, min(samples_per_class, len(right_images)))
    
    # Create list of (image_path, true_label) tuples
    test_images = []
    for img_path in left_sample:
        test_images.append((img_path, 0, "L"))  # 0 = left eye
    for img_path in right_sample:
        test_images.append((img_path, 1, "R"))  # 1 = right eye
        
    # Shuffle the list
    random.shuffle(test_images)
    
    return test_images

# Predict image class, with confidence scores
def predict_image(model, image_path, transform, device):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
    return {
        "predicted_class": predicted_class,
        "predicted_label": "L" if predicted_class == 0 else "R", 
        "confidence": confidence,
        "probabilities": probabilities[0].cpu().numpy(),
        "raw_image": image
    }

# Create a grid of images with predictions
def create_prediction_grid(test_images, predictions, output_dir):
    num_images = len(test_images)
    cols = 4
    rows = (num_images + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    correct_predictions = 0
    
    for idx, ((img_path, true_class, true_label), pred) in enumerate(zip(test_images, predictions)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Display image
        ax.imshow(pred["raw_image"])
        ax.axis('off')
        
        # Check if prediction is correct
        is_correct = pred["predicted_class"] == true_class
        if is_correct:
            correct_predictions += 1
        
        # Create title with prediction info
        color = 'green' if is_correct else 'red'
        title = f"True: {true_label} | Pred: {pred['predicted_label']}\n"
        title += f"Confidence: {pred['confidence']:.3f}"
        
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')
        
        # Add filename as subtitle
        filename = img_path.name
        ax.text(0.5, -0.1, filename, transform=ax.transAxes, 
               ha='center', va='top', fontsize=8, style='italic')
    
    # Hide empty subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    # Add overall accuracy to the title
    accuracy = correct_predictions / num_images
    fig.suptitle(f"Laterality Classification Results - Accuracy: {accuracy:.2%} ({correct_predictions}/{num_images})", 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the grid
    output_path = output_dir / "prediction_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Prediction grid saved to {output_path}")
    return accuracy

# Save detailed results to JSON and print summary statistics
def save_detailed_results(test_images, predictions, output_dir):
    results = []
    
    for (img_path, true_class, true_label), pred in zip(test_images, predictions):
        result = {
            "image_path": str(img_path),
            "filename": img_path.name,
            "true_class": int(true_class),
            "true_label": true_label,
            "predicted_class": int(pred["predicted_class"]),
            "predicted_label": pred["predicted_label"],
            "confidence": float(pred["confidence"]),
            "prob_left": float(pred["probabilities"][0]),
            "prob_right": float(pred["probabilities"][1]),
            "correct": pred["predicted_class"] == true_class
        }
        results.append(result)
    
    # Save to JSON
    results_path = output_dir / "detailed_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to {results_path}")
    
    # Print summary statistics
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total
    
    avg_confidence_correct = np.mean([r["confidence"] for r in results if r["correct"]])
    avg_confidence_incorrect = np.mean([r["confidence"] for r in results if not r["correct"]]) if any(not r["correct"] for r in results) else 0
    
    print(f"\nSummary Statistics:")
    print(f"Total images tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average confidence (correct predictions): {avg_confidence_correct:.3f}")
    print(f"Average confidence (incorrect predictions): {avg_confidence_incorrect:.3f}")


def main():
    args = parse_args()
    
    # Set random seed for reproducible sampling
    random.seed(42)
    torch.manual_seed(42)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, args.backbone, device)
    
    # Get test transform
    transform = get_test_transform()
    
    # Collect test images
    test_images = collect_test_images(args.image_dir, args.num_samples)
    
    if len(test_images) == 0:
        print("No images found for testing. Please check the image directory structure.")
        return
        
    print(f"Selected {len(test_images)} images for testing")
    
    # Make predictions
    print("Making predictions:")
    predictions = []
    for img_path, true_class, true_label in test_images:
        pred = predict_image(model, img_path, transform, device)
        predictions.append(pred)
        print(f"  {img_path.name}: True={true_label}, Pred={pred['predicted_label']} ({pred['confidence']:.3f})")
    
    # Create visualizations
    print("\nCreating visualizations:")
    accuracy = create_prediction_grid(test_images, predictions, args.output_dir)
    
    # Save detailed results
    save_detailed_results(test_images, predictions, args.output_dir)
    
    print(f"\nTesting completed! Results saved to {args.output_dir}")
    

if __name__ == "__main__":
    main()
