#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimCLR (Simple Contrastive Learning) for Corel Dataset

This script implements SimCLR for learning visual representations from the Corel dataset.
SimCLR learns representations through contrastive learning:
- Creates augmented pairs from the same image (positive pairs)
- Uses InfoNCE loss to make positive pairs similar and negative pairs dissimilar
- No need for image reconstruction - focuses purely on representation learning

Prepared for Task 5: Clustering evaluation with self-supervised learning techniques.

Usage:
    python 5A-simclr_corel.py --data-dir training_data/corel/corel_all --output-dir simclr_models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import time
import re
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
try:
    import umap.umap_ as umap
except ImportError:
    try:
        import umap
    except ImportError:
        umap = None
        print("⚠ Warning: umap-learn not installed. UMAP visualization will be disabled.")


class CorelDataset(Dataset):
    """Dataset class for loading Corel images"""
    def __init__(self, data_dir, image_size=224, transform=None):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.data_dir.rglob(ext)))
            self.image_paths.extend(list(self.data_dir.rglob(ext.upper())))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        # Sort for reproducibility
        self.image_paths = sorted(self.image_paths)
        
        # Base transform (no augmentation)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Extract class label from filename (format: XXXX_YYYY.png)
        filename = Path(img_path).name
        match = re.match(r'^(\d+)_\d+\.png$', filename)
        if match:
            class_label = int(match.group(1))
        else:
            class_label = -1  # Unknown class
        
        return image, class_label, str(img_path)


class SimCLREncoder(nn.Module):
    """SimCLR Encoder for RGB images (Corel dataset)"""
    def __init__(self, image_size=224, latent_dim=128):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Encoder backbone (ResNet-like architecture)
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Calculate feature dimension after encoder
        # After 4 stride-2 convolutions: image_size / 16
        # With adaptive pooling: 512
        self.feature_dim = 512
        
        # Projection head for SimCLR (maps to lower dimensional space for contrastive learning)
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        """Forward pass: returns both features and projections"""
        # Get features from encoder
        features = self.encoder(x)
        # Apply projection head
        projections = self.projection_head(features)
        return features, projections
    
    def encode_only(self, x):
        """Extract encoder features only (for representation learning evaluation)"""
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten
        return features


def get_simclr_augmentation(image_size=224):
    """Augmentation pipeline for SimCLR (RGB images)"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def create_simclr_pairs(batch, augment_fn):
    """Create augmented pairs for SimCLR training"""
    batch_size = batch.shape[0]
    
    # Create two augmented versions of each image
    augmented_1 = []
    augmented_2 = []
    
    for img in batch:
        # Convert to proper format for augmentation (denormalize from [-1,1] to [0,1])
        # The image is in format (C, H, W) with values in [-1, 1]
        img_denorm = (img * 0.5 + 0.5).clamp(0, 1)
        
        # augment_fn already includes ToPILImage(), so we pass the tensor directly
        # Create two different augmented versions
        aug1 = augment_fn(img_denorm)
        aug2 = augment_fn(img_denorm)
        
        augmented_1.append(aug1)
        augmented_2.append(aug2)
    
    augmented_1 = torch.stack(augmented_1)
    augmented_2 = torch.stack(augmented_2)
    
    return augmented_1, augmented_2


def simclr_loss(projections_1, projections_2, temperature=0.5):
    """
    SimCLR loss function (InfoNCE/NT-Xent)
    
    Args:
        projections_1: Projections from first augmented batch [batch_size, projection_dim]
        projections_2: Projections from second augmented batch [batch_size, projection_dim]
        temperature: Temperature parameter for softmax
    """
    batch_size = projections_1.shape[0]
    device = projections_1.device
    
    # Normalize projections
    projections_1 = F.normalize(projections_1, dim=1)
    projections_2 = F.normalize(projections_2, dim=1)
    
    # Concatenate projections: [2*batch_size, projection_dim]
    projections = torch.cat([projections_1, projections_2], dim=0)
    
    # Compute similarity matrix: [2*batch_size, 2*batch_size]
    similarity_matrix = torch.matmul(projections, projections.T) / temperature
    
    # Create labels for positive pairs
    # For SimCLR: (i, i+batch_size) and (i+batch_size, i) are positive pairs
    labels = torch.cat([torch.arange(batch_size, 2*batch_size, device=device), 
                       torch.arange(0, batch_size, device=device)])
    
    # Mask to remove self-similarity (diagonal)
    mask = torch.eye(2*batch_size, device=device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss


def train_batch_simclr(data, model, optimizer, augment_fn, temperature=0.5, device='cuda'):
    """Train one batch with SimCLR"""
    model.train()
    data = data.to(device)
    
    # Create augmented pairs
    aug1, aug2 = create_simclr_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass through both augmented versions
    _, proj1 = model(aug1)
    _, proj2 = model(aug2)
    
    # Compute SimCLR loss
    loss = simclr_loss(proj1, proj2, temperature)
    
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def validate_batch_simclr(data, model, augment_fn, temperature=0.5, device='cuda'):
    """Validate one batch with SimCLR"""
    model.eval()
    data = data.to(device)
    
    # Create augmented pairs
    aug1, aug2 = create_simclr_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    
    # Forward pass
    _, proj1 = model(aug1)
    _, proj2 = model(aug2)
    
    # Compute loss
    loss = simclr_loss(proj1, proj2, temperature)
    
    return loss.item()


def extract_features(model, dataloader, device, use_projection=False):
    """Extract features from trained SimCLR model"""
    model.eval()
    features = []
    labels = []
    image_paths = []
    
    print("Extracting features...")
    with torch.no_grad():
        for images, class_labels, paths in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)
            
            if use_projection:
                # Use projection head features
                _, projections = model(images)
                features.append(projections.cpu().numpy())
            else:
                # Use encoder features only (recommended for clustering)
                encoder_features = model.encode_only(images)
                features.append(encoder_features.cpu().numpy())
            
            labels.append(class_labels.numpy())
            image_paths.extend(paths)
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"✓ Extracted features shape: {features.shape}")
    print(f"  Number of images: {len(image_paths)}")
    print(f"  Feature dimension: {features.shape[1]}")
    
    return features, labels, image_paths


def save_features(features, labels, image_paths, output_path, config):
    """Save features in multiple formats"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy array
    np.save(output_path, features)
    print(f"✓ Saved features to: {output_path}")
    
    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    metadata = {
        'num_samples': len(features),
        'feature_dim': features.shape[1],
        'image_size': config.image_size,
        'latent_dim': config.latent_dim,
        'num_classes': len(set(labels)) if len(set(labels)) > 1 else None,
        'method': 'SimCLR',
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")
    
    # Save labels
    labels_path = output_path.with_suffix('.labels.npy')
    np.save(labels_path, labels)
    print(f"✓ Saved labels to: {labels_path}")
    
    # Save image paths
    paths_path = output_path.with_suffix('.paths.txt')
    with open(paths_path, 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")
    print(f"✓ Saved image paths to: {paths_path}")


def evaluate_clustering(features, labels, output_dir):
    """Evaluate clustering quality"""
    print("\n" + "="*60)
    print("CLUSTERING EVALUATION")
    print("="*60)
    
    # Get number of classes
    unique_labels = np.unique(labels[labels >= 0])  # Exclude -1 (unknown)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        print("⚠ Not enough classes for clustering evaluation")
        return None
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-means clustering
    print(f"\nPerforming K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(features_scaled)
    
    # Calculate metrics
    valid_mask = labels >= 0
    if valid_mask.sum() > 0:
        ari = adjusted_rand_score(labels[valid_mask], predicted_labels[valid_mask])
        nmi = normalized_mutual_info_score(labels[valid_mask], predicted_labels[valid_mask])
        
        # Silhouette score (on scaled features)
        silhouette = silhouette_score(features_scaled, predicted_labels)
        
        print(f"\nClustering Metrics:")
        print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
        print(f"  Silhouette Score: {silhouette:.4f}")
        
        metrics = {
            'ari': float(ari),
            'nmi': float(nmi),
            'silhouette': float(silhouette),
            'n_clusters': int(n_clusters),
        }
        
        # Save metrics
        metrics_path = Path(output_dir) / 'clustering_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Saved metrics to: {metrics_path}")
        
        return metrics
    else:
        print("⚠ No valid labels found for evaluation")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Train SimCLR for Corel Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SimCLR
  python 5A-simclr_corel.py --data-dir training_data/corel/corel_all --output-dir simclr_models
  
  # Train and extract features
  python 5A-simclr_corel.py --data-dir training_data/corel/corel_all \\
      --output-dir simclr_models --extract-features features/simclr_features.npy
  
  # Custom parameters
  python 5A-simclr_corel.py --data-dir training_data/corel/corel_all \\
      --epochs 100 --batch-size 32 --latent-dim 256
        """
    )
    
    # Path arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='simclr_models',
        help='Output directory for SimCLR model (default: simclr_models)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='.',
        help='Base directory for paths (default: current directory)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image size for training (default: 224)'
    )
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=128,
        help='Latent dimension for projection head (default: 128)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-3,
        help='Learning rate (default: 3e-3)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5,
        help='Temperature parameter for SimCLR loss (default: 0.5)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay (default: 1e-4)'
    )
    
    # Feature extraction arguments
    parser.add_argument(
        '--extract-features',
        type=str,
        default=None,
        help='Extract features after training (output path for .npy file)'
    )
    parser.add_argument(
        '--use-projection',
        action='store_true',
        help='Use projection head features instead of encoder features'
    )
    parser.add_argument(
        '--evaluate-clustering',
        action='store_true',
        help='Evaluate clustering quality after feature extraction'
    )
    
    # Other arguments
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of DataLoader workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir).resolve()
    data_dir = base_dir / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    data_dir = data_dir.resolve()
    output_dir = base_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify CUDA availability
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"\n{'='*60}")
        print("CUDA AVAILABLE - GPU OPTIMIZATION ENABLED")
        print(f"{'='*60}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Device: {device}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"{'='*60}\n")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        device = "cpu"
        print("\n⚠ WARNING: CUDA not available! Training will be VERY slow on CPU.")
        print("⚠ This script is optimized for GPU.\n")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("="*60)
    print("SIMCLR TRAINING FOR COREL DATASET")
    print("="*60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Base directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print("="*60 + "\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = CorelDataset(str(data_dir), image_size=args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.startswith("cuda") else False,
        drop_last=True
    )
    
    print(f"✓ Dataset: {len(dataset)} images")
    
    # Count classes
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    unique_classes = set([l for l in all_labels if l >= 0])
    print(f"✓ Classes found: {len(unique_classes)}\n")
    
    # Create model
    model = SimCLREncoder(image_size=args.image_size, latent_dim=args.latent_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {num_params:,}\n")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Augmentation function
    augment_fn = get_simclr_augmentation(args.image_size)
    
    # Training state
    best_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        print(f"✓ Resumed from epoch {start_epoch}\n")
    
    # Training loop
    print("="*60)
    print("TRAINING...")
    print("="*60 + "\n")
    
    training_start_time = time.time()
    losses = []
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Training
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, _, _ in pbar:
            loss = train_batch_simclr(
                images, model, optimizer, augment_fn,
                temperature=args.temperature, device=device
            )
            epoch_losses.append(loss)
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Epoch Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses,
                'config': {
                    'image_size': args.image_size,
                    'latent_dim': args.latent_dim,
                    'temperature': args.temperature,
                }
            }, Path(output_dir) / 'best_model.pt')
            print(f"  ✓ Saved best model")
    
    total_time = time.time() - training_start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Total training time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print("="*60)
    
    # Extract features if requested
    if args.extract_features:
        print("\n" + "="*60)
        print("EXTRACTING FEATURES...")
        print("="*60)
        
        # Create dataset without augmentation for feature extraction
        eval_dataset = CorelDataset(str(data_dir), image_size=args.image_size)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.startswith("cuda") else False,
        )
        
        features, labels, image_paths = extract_features(
            model, eval_dataloader, device, use_projection=args.use_projection
        )
        
        # Save features
        output_features_path = base_dir / args.extract_features if not Path(args.extract_features).is_absolute() else Path(args.extract_features)
        save_features(features, labels, image_paths, str(output_features_path), args)
        
        # Evaluate clustering if requested
        if args.evaluate_clustering:
            evaluate_clustering(features, labels, output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
