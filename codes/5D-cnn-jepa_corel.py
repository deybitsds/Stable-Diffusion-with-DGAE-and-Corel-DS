#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN-JEPA (Joint Embedding Predictive Architecture) for Corel Dataset

This script implements CNN-JEPA for learning visual representations from the Corel dataset.
CNN-JEPA learns representations by predicting embeddings from one view to another without
explicit data augmentation, using a joint embedding predictive architecture.

Based on: CNN-JEPA: Self-Supervised Pretraining Convolutional Neural Networks Using 
Joint Embedding Predictive Architecture (arXiv:2408.07514)

Prepared for Task 5: Clustering evaluation with self-supervised learning techniques.

Usage:
    python 5D-cnn-jepa_corel.py --data-dir training_data/corel/corel_all --output-dir cnn_jepa_models
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
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans


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


class CNNEncoder(nn.Module):
    """CNN Encoder for CNN-JEPA"""
    def __init__(self, image_size=224, feature_dim=512):
        super().__init__()
        self.image_size = image_size
        self.feature_dim = feature_dim
        
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
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
        )
    
    def forward(self, x):
        """Extract features and project"""
        features = self.encoder(x)
        projected = self.projection(features)
        return projected
    
    def encode_only(self, x):
        """Extract encoder features only (for representation learning evaluation)"""
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten
        return features


class Predictor(nn.Module):
    """Predictor network for CNN-JEPA"""
    def __init__(self, feature_dim=512, hidden_dim=512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, feature_dim),
        )
    
    def forward(self, x):
        return self.predictor(x)


class CNNJEPA(nn.Module):
    """CNN-JEPA: Joint Embedding Predictive Architecture"""
    def __init__(self, image_size=224, feature_dim=512, hidden_dim=512):
        super().__init__()
        self.encoder = CNNEncoder(image_size, feature_dim)
        self.predictor = Predictor(feature_dim, hidden_dim)
        self.feature_dim = feature_dim
    
    def forward(self, x_context, x_target):
        """
        Forward pass for CNN-JEPA
        
        Args:
            x_context: Context view of the image
            x_target: Target view of the image
        """
        # Encode both views
        z_context = self.encoder(x_context)
        z_target = self.encoder(x_target)
        
        # Predict target embedding from context
        z_pred = self.predictor(z_context)
        
        return z_context, z_target, z_pred
    
    def encode_only(self, x):
        """Extract encoder features only"""
        return self.encoder.encode_only(x)


def get_jepa_views(image_size=224):
    """Create two different views of the image for CNN-JEPA"""
    # View 1: Random crop and resize
    view1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # View 2: Different random crop and resize
    view2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return view1, view2


def create_jepa_views(batch, view1_fn, view2_fn):
    """Create context and target views for CNN-JEPA"""
    context_views = []
    target_views = []
    
    for img in batch:
        # Convert to proper format (denormalize from [-1,1] to [0,1])
        img_denorm = (img * 0.5 + 0.5).clamp(0, 1)
        
        # Create two different views
        view1 = view1_fn(img_denorm)
        view2 = view2_fn(img_denorm)
        
        context_views.append(view1)
        target_views.append(view2)
    
    context_views = torch.stack(context_views)
    target_views = torch.stack(target_views)
    
    return context_views, target_views


def jepa_loss(z_context, z_target, z_pred, temperature=0.1):
    """
    CNN-JEPA loss function
    
    The loss encourages the predicted embedding to match the target embedding
    while keeping embeddings normalized.
    """
    # Normalize embeddings
    z_target = F.normalize(z_target, dim=1)
    z_pred = F.normalize(z_pred, dim=1)
    
    # Cosine similarity loss (negative cosine similarity)
    loss = -F.cosine_similarity(z_pred, z_target, dim=1).mean()
    
    # Alternative: MSE loss on normalized embeddings
    # loss = F.mse_loss(z_pred, z_target)
    
    return loss


def train_batch_jepa(data, model, optimizer, view1_fn, view2_fn, device='cuda'):
    """Train one batch with CNN-JEPA"""
    model.train()
    data = data.to(device)
    
    # Create context and target views
    x_context, x_target = create_jepa_views(data, view1_fn, view2_fn)
    x_context, x_target = x_context.to(device), x_target.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass
    z_context, z_target, z_pred = model(x_context, x_target)
    
    # Compute JEPA loss
    loss = jepa_loss(z_context, z_target, z_pred)
    
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def validate_batch_jepa(data, model, view1_fn, view2_fn, device='cuda'):
    """Validate one batch with CNN-JEPA"""
    model.eval()
    data = data.to(device)
    
    # Create context and target views
    x_context, x_target = create_jepa_views(data, view1_fn, view2_fn)
    x_context, x_target = x_context.to(device), x_target.to(device)
    
    # Forward pass
    z_context, z_target, z_pred = model(x_context, x_target)
    
    # Compute loss
    loss = jepa_loss(z_context, z_target, z_pred)
    
    return loss.item()


def extract_features(model, dataloader, device):
    """Extract features from trained CNN-JEPA model"""
    model.eval()
    features = []
    labels = []
    image_paths = []
    
    print("Extracting features...")
    with torch.no_grad():
        for images, class_labels, paths in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)
            
            # Use encoder features only
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
        'feature_dim': config.feature_dim,
        'num_classes': len(set(labels)) if len(set(labels)) > 1 else None,
        'method': 'CNN-JEPA',
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
        description='Train CNN-JEPA for Corel Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train CNN-JEPA
  python 5D-cnn-jepa_corel.py --data-dir training_data/corel/corel_all --output-dir cnn_jepa_models
  
  # Train and extract features
  python 5D-cnn-jepa_corel.py --data-dir training_data/corel/corel_all \\
      --output-dir cnn_jepa_models --extract-features features/cnn_jepa_features.npy
  
  # Custom parameters
  python 5D-cnn-jepa_corel.py --data-dir training_data/corel/corel_all \\
      --epochs 100 --batch-size 32 --feature-dim 256
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
        default='cnn_jepa_models',
        help='Output directory for CNN-JEPA model (default: cnn_jepa_models)'
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
        '--feature-dim',
        type=int,
        default=512,
        help='Feature dimension (default: 512)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=512,
        help='Hidden dimension for predictor (default: 512)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
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
    print("CNN-JEPA TRAINING FOR COREL DATASET")
    print("="*60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Base directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Feature dimension: {args.feature_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
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
    model = CNNJEPA(
        image_size=args.image_size,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {num_params:,}\n")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # View functions
    view1_fn, view2_fn = get_jepa_views(args.image_size)
    
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
            loss = train_batch_jepa(
                images, model, optimizer, view1_fn, view2_fn, device=device
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
                    'feature_dim': args.feature_dim,
                    'hidden_dim': args.hidden_dim,
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
            model, eval_dataloader, device
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

