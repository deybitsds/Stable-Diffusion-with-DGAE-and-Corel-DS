#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract Latent Features from DGAE for Clustering Evaluation

This script extracts latent features from a trained DGAE model for clustering evaluation.
The features are saved in a format suitable for clustering algorithms (e.g., k-means, hierarchical clustering).

Usage:
    python 4B-extract-features-corel.py --model-checkpoint dgae_models/best_model.pt --data-dir training_data/corel/corel_all --output features/dgae_features.npy
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
import pickle
import json
from collections import defaultdict


# Import DGAE model components (same as 4A)
class Config:
    """Configuration class - must match training config"""
    image_size = 256
    image_channels = 3
    latent_dim = 128
    hidden_dims = [64, 128, 256, 512]


class ResidualBlock(nn.Module):
    """Simple residual block"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class Encoder(nn.Module):
    """Encoder for DGAE"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        layers = []
        in_channels = config.image_channels
        
        for h_dim in config.hidden_dims:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2),
                ResidualBlock(h_dim),
            ))
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        self.final_size = config.image_size // (2 ** len(config.hidden_dims))
        self.final_channels = config.hidden_dims[-1]
        flatten_dim = self.final_channels * self.final_size * self.final_size
        
        self.fc = nn.Linear(flatten_dim, config.latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class SimpleDataset(Dataset):
    """Dataset class for loading images"""
    def __init__(self, data_dir, image_size):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.data_dir.rglob(ext)))
            self.image_paths.extend(list(self.data_dir.rglob(ext.upper())))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        # Sort for reproducibility
        self.image_paths = sorted(self.image_paths)
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), str(img_path)


def load_dgae_model(checkpoint_path, device):
    """Load DGAE model from checkpoint"""
    print(f"Loading DGAE model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint or use defaults
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        if isinstance(saved_config, dict):
            config = Config()
            config.latent_dim = saved_config.get('latent_dim', config.latent_dim)
            config.image_size = saved_config.get('image_size', config.image_size)
            config.image_channels = saved_config.get('image_channels', config.image_channels)
            config.hidden_dims = saved_config.get('hidden_dims', config.hidden_dims)
        else:
            config = saved_config
    else:
        config = Config()
    
    # Create encoder (we only need encoder for feature extraction)
    encoder = Encoder(config).to(device)
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    
    # Extract encoder weights (handle both 'encoder.' prefix and direct keys)
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            new_key = key[8:]  # Remove 'encoder.' prefix
            encoder_state_dict[new_key] = value
        elif not key.startswith('decoder.'):
            # If no prefix, assume it's encoder weights
            encoder_state_dict[key] = value
    
    encoder.load_state_dict(encoder_state_dict, strict=False)
    encoder.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Latent dimension: {config.latent_dim}")
    print(f"  Image size: {config.image_size}")
    
    return encoder, config


def extract_features(encoder, dataloader, device):
    """Extract latent features from all images"""
    encoder.eval()
    features = []
    image_paths = []
    
    print("\nExtracting features...")
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)
            z = encoder(images)
            features.append(z.cpu().numpy())
            image_paths.extend(paths)
    
    features = np.concatenate(features, axis=0)
    print(f"✓ Extracted features shape: {features.shape}")
    print(f"  Number of images: {len(image_paths)}")
    print(f"  Feature dimension: {features.shape[1]}")
    
    return features, image_paths


def get_class_labels(image_paths):
    """Extract class labels from image paths (format: XXXX_YYYY.png)"""
    import re
    pattern = re.compile(r'(\d+)_\d+\.png$')
    
    labels = []
    class_mapping = {}
    
    for img_path in image_paths:
        img_name = Path(img_path).name
        match = pattern.match(img_name)
        if match:
            class_num = int(match.group(1))
            labels.append(class_num)
            if class_num not in class_mapping:
                class_mapping[class_num] = len(class_mapping)
        else:
            labels.append(-1)  # Unknown class
    
    return np.array(labels), class_mapping


def save_features(features, image_paths, labels, output_path, config):
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
        'class_mapping': {str(k): v for k, v in labels[1].items()} if isinstance(labels, tuple) else None,
        'num_classes': len(set(labels[0])) if isinstance(labels, tuple) else None,
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")
    
    # Save labels if available
    if isinstance(labels, tuple):
        labels_array, class_mapping = labels
        labels_path = output_path.with_suffix('.labels.npy')
        np.save(labels_path, labels_array)
        print(f"✓ Saved labels to: {labels_path}")
        
        # Save class mapping
        mapping_path = output_path.with_suffix('.class_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump({str(k): int(v) for k, v in class_mapping.items()}, f, indent=2)
        print(f"✓ Saved class mapping to: {mapping_path}")
    
    # Save image paths
    paths_path = output_path.with_suffix('.paths.txt')
    with open(paths_path, 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")
    print(f"✓ Saved image paths to: {paths_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract Latent Features from DGAE for Clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features from trained DGAE
  python 4B-extract-features-corel.py \\
      --model-checkpoint dgae_models/best_model.pt \\
      --data-dir training_data/corel/corel_all \\
      --output features/dgae_features.npy
  
  # Extract with custom batch size
  python 4B-extract-features-corel.py \\
      --model-checkpoint dgae_models/best_model.pt \\
      --data-dir training_data/corel/corel_all \\
      --output features/dgae_features.npy \\
      --batch-size 32
        """
    )
    
    # Path arguments
    parser.add_argument(
        '--model-checkpoint',
        type=str,
        required=True,
        help='Path to DGAE model checkpoint (best_model.pt)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for features (.npy file)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='.',
        help='Base directory for paths (default: current directory)'
    )
    
    # Extraction arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for feature extraction (default: 32)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of DataLoader workers (default: 4)'
    )
    parser.add_argument(
        '--extract-labels',
        action='store_true',
        help='Extract class labels from image filenames (format: XXXX_YYYY.png)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir).resolve()
    model_checkpoint = base_dir / args.model_checkpoint if not Path(args.model_checkpoint).is_absolute() else Path(args.model_checkpoint)
    model_checkpoint = model_checkpoint.resolve()
    data_dir = base_dir / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    data_dir = data_dir.resolve()
    output_path = base_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output_path = output_path.resolve()
    
    # Verify CUDA availability
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"\n{'='*60}")
        print("CUDA AVAILABLE - GPU OPTIMIZATION ENABLED")
        print(f"{'='*60}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")
    else:
        device = "cpu"
        print("\n⚠ WARNING: CUDA not available! Extraction will be slower on CPU.\n")
    
    print("="*60)
    print("DGAE FEATURE EXTRACTION FOR CLUSTERING")
    print("="*60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Base directory: {base_dir}")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Data directory: {data_dir}")
    print(f"Output path: {output_path}")
    print(f"Batch size: {args.batch_size}")
    print("="*60 + "\n")
    
    # Verify files exist
    if not model_checkpoint.exists():
        print(f"ERROR: Model checkpoint not found: {model_checkpoint}")
        return 1
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1
    
    # Load model
    encoder, config = load_dgae_model(str(model_checkpoint), device)
    
    # Load dataset
    dataset = SimpleDataset(str(data_dir), config.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle for consistent ordering
        num_workers=args.num_workers,
        pin_memory=True if device.startswith("cuda") else False,
    )
    
    print(f"✓ Dataset: {len(dataset)} images\n")
    
    # Extract features
    features, image_paths = extract_features(encoder, dataloader, device)
    
    # Extract labels if requested
    labels = None
    if args.extract_labels:
        labels = get_class_labels(image_paths)
        print(f"✓ Extracted labels for {len(set(labels[0]))} classes")
    
    # Save features
    save_features(features, image_paths, labels, str(output_path), config)
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*60)
    print(f"Features saved to: {output_path}")
    print(f"Shape: {features.shape}")
    print(f"Use these features for clustering evaluation (k-means, hierarchical, etc.)")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

