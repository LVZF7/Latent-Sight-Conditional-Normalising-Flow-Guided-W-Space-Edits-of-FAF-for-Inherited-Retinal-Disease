"""
Objective: Train a CNN to classify real FAF images into gene_0 vs gene_35.

Expected folder structure:
    data/
        geneCNN_training_splits/
            train/
                gene_0/*.png
                gene_35/*.png
            val/
                gene_0/*.png
                gene_35/*.png
            test/
                gene_0/*.png
                gene_35/*.png

To Run:
    python scripts/gene_detector/train_gene_cnn.py \
        --data_root data/real/geneCNN_training_splits \
        --output_dir weights/gene_120epoch \
        --epochs 120 \
        --batch-size 64 \
        --backbone resnet18 \
        --learning-rate 1e-4 \
        --weight-decay 1e-4 \
        --plateau-patience 8 \
        --early-stop-patience 20 \
        --lr-factor 0.3 \
        --min-lr 1e-6 \
        --device cuda \
        --num-workers 4 \
        --seed 42
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Argument parsing dedicated function
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--backbone", type=str, default="resnet18")

    # training
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # sched & early stop
    p.add_argument("--plateau-patience", type=int, default=8)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--lr-factor", type=float, default=0.3)
    p.add_argument("--min-lr", type=float, default=1e-6)

    # system
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Set up CUDA or CPU device
def device_setup(device):
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")
    return device


# Check that the folder used for training has the expected structure
def validate_data_structure(data_root):
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory does not exist: {data_root}")
    
    target_splits = ["train", "val", "test"]
    target_classes = ["gene_0", "gene_35"]

    for split in target_splits:
        split_dir = data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"{split} does not exist in {split_dir}")

        for cls in target_classes:
            class_dir = split_dir / cls
            if not class_dir.exists():
                raise FileNotFoundError(f"{cls} does not exist in {class_dir}")
            
            # check if there are images in the L/R folders
            images = list(class_dir.rglob("*.png"))
            if not images:
                logger.warning(f"No PNG images found in {class_dir}")

             
# Data transforms for training and validation sets
def get_data_transforms():
    # normalization values for ImageNet (see wiki: https://en.wikipedia.org/wiki/ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # based on torchvision's default transforms for ResNet
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform


# Create DataLoader for training and validation sets
def create_dataloaders(data_root, batch_size, num_workers):
    train_transform, val_transform = get_data_transforms()
    
    train_dataset = datasets.ImageFolder(data_root / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_root / "val", transform=val_transform)
    test_dataset = datasets.ImageFolder(data_root / "test", transform=val_transform)
    
    # log dataset sizes
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    logger.info(f"Class Mapping: {train_dataset.class_to_idx}")

    # verify class mapping is gene_0=0, gene_35=1
    if train_dataset.class_to_idx != {'gene_0': 0, 'gene_35': 1}:
        logger.warning(f"Unexpected class mapping: {train_dataset.class_to_idx}")
        logger.warning("Expected: {{'gene_0': 0, 'gene_35': 1}}")
        
    # dataloaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    return trainloader, valloader, testloader, train_dataset.class_to_idx

# Create model based on specified backbone
def create_model(backbone):
    if backbone.lower() == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)  # 2 classes: gene_0 and gene_35
    else:
        raise ValueError(f"Unsupported backbone model: {backbone}. Supported: resnet18")

    logger.info(f"Created {backbone} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# run epoch of training or validation
def run_epoch(model, dataloader, criterion, optimiser, device, is_training, class_to_idx):
    model.train() if is_training else model.eval()
    
    running_loss = 0.0
    accuracy_metric = BinaryAccuracy().to(device)
    auc_metric = BinaryAUROC().to(device)

    gene_35_class_idx = class_to_idx['gene_35']  # Use gene_35 as positive class for binary metrics

    with torch.set_grad_enabled(is_training):
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            
            #forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                
            # update metrics
            running_loss += loss.item() * images.size(0)
            probabilities = torch.softmax(logits, dim=1)[:, gene_35_class_idx]  # probability of gene_35 class
            binary_labels = (labels == gene_35_class_idx).float()  # Convert to binary: 1 if gene_35, 0 if gene_0
            accuracy_metric.update(probabilities, binary_labels)
            auc_metric.update(probabilities, binary_labels)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = accuracy_metric.compute().item()
    epoch_auc = auc_metric.compute().item()

    return {"loss": epoch_loss, "acc": epoch_accuracy, "auc": epoch_auc}


# Save model checkpoint with metadata
def save_checkpoint(model, optimizer, epoch, metrics, output_dir, class_to_idx, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "class_to_idx": class_to_idx,
        "timestamp": time.time(),
    }
    
    checkpoint_path = output_dir / f"epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = output_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model checkpoint to {best_path} with AUC {round(metrics['auc'], 4)}")

# Plot training curves
def plot_training_curves(training_history, test_metrics, output_dir):
    epochs = [entry['epoch'] for entry in training_history]

    # training and validation losses
    train_losses = [entry['train']['loss'] for entry in training_history]
    val_losses = [entry['val']['loss'] for entry in training_history]
    
    # validation accuracy and AUC
    val_accuracies = [entry['val']['acc'] for entry in training_history]
    val_aucs = [entry['val']['auc'] for entry in training_history]
    
    # create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Training Curves", fontsize=16)
    
    # Loss curves
    axs[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
    axs[0, 0].plot(epochs, val_losses, label='Val Loss', color='orange')
    axs[0, 0].set_title("Loss Curve")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Accuracy curve
    axs[0, 1].plot(epochs, val_accuracies, label='Val Accuracy', color='green')
    axs[0, 1].set_title("Validation Accuracy Curve")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # AUC curve
    axs[1, 0].plot(epochs, val_aucs, label='Val AUC', color='purple')
    axs[1, 0].set_title("Validation AUC Curve")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("AUC")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 0].set_ylim(0, 1)  # AUC is between 0 and 1
    
    # Test results in bottom right
    axs[1, 1].text(0.1, 0.7, f"Final Test Results:", fontsize=14, weight='bold')
    axs[1, 1].text(0.1, 0.5, f"Accuracy: {test_metrics['acc']:.3f}", fontsize=12)
    axs[1, 1].text(0.1, 0.3, f"AUC: {test_metrics['auc']:.4f}", fontsize=12)
    axs[1, 1].text(0.1, 0.1, f"Loss: {test_metrics['loss']:.4f}", fontsize=12)
    axs[1, 1].set_xlim(0, 1)
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plot_save_path = output_dir / "training_curves.png"
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Training curves saved to {plot_save_path}")
    
    
def main():
    args = parse_args()
    
    # random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Set up device
    device = device_setup(args.device)
    
    # Ensure output directory is under weights/ and create it
    if not str(args.output_dir).startswith('weights/'):
        base_name = args.output_dir.name
        args.output_dir = Path('weights') / f"{base_name}_{args.epochs}epoch"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # set up training log
    log_file = args.output_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"Starting gene training with args: {vars(args)}")
    
    # Validate data structure and make dataloaders
    validate_data_structure(args.data_root)
    trainloader, valloader, testloader, class_to_idx = create_dataloaders(args.data_root, args.batch_size, args.num_workers)
    
    # create model, loss function, optimiser
    model = create_model(args.backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # scheduler for learning rate decay on plateau of validation AUC
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='max', factor=args.lr_factor,
        patience=args.plateau_patience, threshold=0.001,
        threshold_mode='abs', min_lr=args.min_lr, verbose=True
    )
    
    # training loop
    best_auc = 0.0
    epochs_since_best = 0
    training_history = []
    
    logger.info("Starting training")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # train and validate
        train_metrics = run_epoch(model, trainloader, criterion, optimiser, device, is_training=True, class_to_idx=class_to_idx)
        val_metrics = run_epoch(model, valloader, criterion, None, device, is_training=False, class_to_idx=class_to_idx)
        
        # Scheduler step on validation AUC
        scheduler.step(val_metrics['auc'])
        
        # Track best
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            epochs_since_best = 0
            is_best = True
        else:
            epochs_since_best += 1
            is_best = False
            
        save_checkpoint(model, optimiser, epoch, val_metrics, args.output_dir, class_to_idx, is_best=is_best)
        
        # Early stopping
        if epochs_since_best >= args.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch} (no AUC improvement for {epochs_since_best} epochs).")
            break
        
        # log metrics from epoch
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} ({epoch_time:.1f}s) - "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.3f}, "
            f"AUC: {val_metrics['auc']:.4f}"
        )
        
        # log current learning rate (in case its been adjusted)
        current_lr = optimiser.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.6f}")
        
        # record training history
        training_history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "epoch_time": epoch_time
        })
        
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f} seconds")
    
    # Final test evaluation using BEST model (not last)
    logger.info("Evaluating on test set using BEST model")
    best_checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, testloader, criterion, None, device, is_training=False, class_to_idx=class_to_idx)
    
    logger.info(
        f"BEST Model Test Results - Loss: {test_metrics['loss']:.4f}, "
        f"Accuracy: {test_metrics['acc']:.3f}, AUC: {test_metrics['auc']:.4f}"
    )

    plot_training_curves(training_history, test_metrics, args.output_dir)
    
    # save summary of training
    summary_path = args.output_dir / "training_summary.json"
    with summary_path.open("w") as f:
        # Convert PosixPath objects to strings for JSON 
        args_dict = vars(args).copy()
        args_dict['data_root'] = str(args_dict['data_root'])
        args_dict['output_dir'] = str(args_dict['output_dir'])

        json.dump({
            "args": args_dict,
            "class_to_idx": class_to_idx,
            "best_val_auc": best_auc,
            "test_metrics": test_metrics,
            "training_history": training_history,
            "total_training_time": total_time
        }, f, indent=2)
        
    logger.info(f"Training summary saved to {summary_path}")
    logger.info(f"Best model saved to: {args.output_dir / 'best_model.pt'}")
    logger.info("Training complete")

if __name__ == "__main__":
    main()