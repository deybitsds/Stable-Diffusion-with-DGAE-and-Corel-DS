#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train DGAE (Diffusion-Guided Autoencoder) for Corel Dataset

This script implements DGAE: Diffusion-Guided Autoencoder for Efficient Latent Representation Learning.
It trains a convolutional autoencoder guided by a pre-trained Stable Diffusion model with LoRA weights
from Task 2. The goal is to learn latent features for clustering, NOT for data augmentation.

Based on the DGAE paper: https://arxiv.org/abs/2506.09644

Usage:
    python 4A-train-dgae-corel.py --data-dir training_data/corel/corel_all --lora-dir corel_models --output-dir dgae_models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import argparse
import sys
import time
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.utils import make_image_grid
import gc


class Config:
    """Configuration class for DGAE training"""
    image_size = 256  # DGAE typically uses 256x256
    image_channels = 3
    latent_dim = 128  # Compact latent space (2x smaller than typical VAE)
    hidden_dims = [64, 128, 256, 512]
    num_epochs = 300
    batch_size = 8  # Smaller batch due to diffusion model memory
    learning_rate = 1e-4
    
    # DGAE-specific: Diffusion guidance weight
    diffusion_guidance_weight = 0.1  # Weight for diffusion-guided loss
    
    # Reconstruction loss
    recon_weight = 1.0
    
    # Perceptual loss (optional, for sharpness)
    use_perceptual = True
    perceptual_weight = 0.03
    
    weight_decay = 1e-5
    grad_clip = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    save_every = 20
    sample_every = 10
    seed = 42
    
    # Diffusion model settings
    diffusion_steps = 50  # Number of diffusion steps for guidance
    guidance_scale = 7.5


class SimpleDataset(Dataset):
    """Dataset class for loading images from directory"""
    def __init__(self, data_dir, image_size):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.data_dir.rglob(ext)))
            self.image_paths.extend(list(self.data_dir.rglob(ext.upper())))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"✓ Found {len(self.image_paths)} images")
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for sharpness"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = vgg.features[:16].eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        x = (x + 1) / 2
        return (x - self.mean) / self.std
    
    def forward(self, x, y):
        x_norm = self.normalize(x)
        y_norm = self.normalize(y)
        x_features = self.feature_extractor(x_norm)
        y_features = self.feature_extractor(y_norm)
        return F.mse_loss(x_features, y_features)


class ResidualBlock(nn.Module):
    """Simple residual block for better features"""
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
    """Encoder for DGAE - compresses images to latent space"""
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
        
        # DGAE uses deterministic encoding (no variational component)
        self.fc = nn.Linear(flatten_dim, config.latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class Decoder(nn.Module):
    """Decoder for DGAE - reconstructs from latent space"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.final_size = config.image_size // (2 ** len(config.hidden_dims))
        self.final_channels = config.hidden_dims[-1]
        
        self.decoder_input = nn.Linear(config.latent_dim, 
                                       self.final_channels * self.final_size * self.final_size)
        
        layers = []
        reversed_dims = list(reversed(config.hidden_dims))
        
        for i in range(len(reversed_dims) - 1):
            layers.append(nn.Sequential(
                ResidualBlock(reversed_dims[i]),
                nn.Conv2d(reversed_dims[i], reversed_dims[i+1] * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(reversed_dims[i+1]),
                nn.LeakyReLU(0.2),
            ))
        
        layers.append(nn.Sequential(
            ResidualBlock(reversed_dims[-1]),
            nn.Conv2d(reversed_dims[-1], config.image_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Tanh()
        ))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.final_channels, self.final_size, self.final_size)
        return self.decoder(h)


class DGAE(nn.Module):
    """Diffusion-Guided Autoencoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        if config.use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def encode(self, x):
        """Encode image to latent representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to image"""
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def load_diffusion_model(lora_dir, lora_name, pretrained_model, device):
    """Load Stable Diffusion model with LoRA weights"""
    print(f"\nLoading Stable Diffusion model with LoRA...")
    print(f"  Base model: {pretrained_model}")
    print(f"  LoRA directory: {lora_dir}")
    print(f"  LoRA file: {lora_name}")
    
    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    # Enable memory optimizations
    if torch.cuda.is_available():
        pipe.enable_attention_slicing(slice_size=1)
        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()
    
    # Load LoRA weights
    lora_path = Path(lora_dir)
    pipe.load_lora_weights(
        pretrained_model_name_or_path_or_dict=str(lora_path),
        weight_name=lora_name,
        adapter_name="corel_lora"
    )
    pipe.set_adapters(["corel_lora"], adapter_weights=[1.0])
    
    print("✓ Diffusion model loaded successfully")
    return pipe


def diffusion_guidance_loss(pipe, recon_images, original_images, config, device):
    """
    Compute diffusion-guided loss using the pre-trained diffusion model.
    
    The idea is to use the diffusion model to refine the reconstruction,
    guiding the decoder to recover informative signals.
    """
    # Convert images from [-1, 1] to [0, 1] for VAE encoding
    # Stable Diffusion VAE expects images in [0, 1] range
    recon_normalized = (recon_images + 1) / 2
    original_normalized = (original_images + 1) / 2
    
    # Clamp to valid range
    recon_normalized = torch.clamp(recon_normalized, 0, 1)
    original_normalized = torch.clamp(original_normalized, 0, 1)
    
    # Resize to 512x512 if needed (Stable Diffusion VAE expects 512)
    if recon_normalized.shape[-1] != 512:
        recon_normalized = F.interpolate(recon_normalized, size=(512, 512), mode='bilinear', align_corners=False)
        original_normalized = F.interpolate(original_normalized, size=(512, 512), mode='bilinear', align_corners=False)
    
    # Use a simple prompt for Corel images
    prompt = "a photo of a corel image"
    
    # Encode images to latent space using VAE
    with torch.no_grad():
        # Encode original images
        original_latents = pipe.vae.encode(original_normalized).latent_dist.sample()
        original_latents = original_latents * pipe.vae.config.scaling_factor
        
        # Encode reconstructed images
        recon_latents = pipe.vae.encode(recon_normalized).latent_dist.sample()
        recon_latents = recon_latents * pipe.vae.config.scaling_factor
    
    # Get text embeddings
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            [prompt] * recon_images.shape[0],
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        ).to(device)
        text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]
    
    # Sample random timesteps
    timesteps = torch.randint(
        0, pipe.scheduler.config.num_train_timesteps,
        (recon_images.shape[0],),
        device=device
    ).long()
    
    # Add noise to latents
    noise = torch.randn_like(original_latents)
    noisy_latents = pipe.scheduler.add_noise(original_latents, noise, timesteps)
    
    # Predict noise using UNet
    with torch.no_grad():
        # Use original latents as target
        model_pred_original = pipe.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Use reconstructed latents
        noisy_recon_latents = pipe.scheduler.add_noise(recon_latents, noise, timesteps)
        model_pred_recon = pipe.unet(
            noisy_recon_latents,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
    
    # Diffusion guidance loss: encourage reconstructed latents to produce similar noise predictions
    # This guides the decoder to learn representations that are compatible with the diffusion model
    diffusion_loss = F.mse_loss(model_pred_recon, model_pred_original, reduction='mean')
    
    return diffusion_loss


def dgae_loss(recon_x, x, z, model, diffusion_pipe, config, device):
    """DGAE loss: reconstruction + perceptual + diffusion guidance"""
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # Perceptual loss
    if model.config.use_perceptual:
        perceptual = model.perceptual_loss(recon_x, x)
        recon_loss = recon_loss + model.config.perceptual_weight * perceptual
    else:
        perceptual = torch.tensor(0.0)
    
    # Diffusion guidance loss (computed less frequently to save memory)
    diffusion_loss = torch.tensor(0.0)
    if diffusion_pipe is not None:
        try:
            # Only compute diffusion loss every few batches to save memory
            diffusion_loss = diffusion_guidance_loss(diffusion_pipe, recon_x, x, config, device)
        except Exception as e:
            print(f"⚠ Warning: Diffusion guidance loss computation failed: {e}")
            diffusion_loss = torch.tensor(0.0)
    
    # Total loss
    total_loss = (config.recon_weight * recon_loss + 
                  config.diffusion_guidance_weight * diffusion_loss)
    
    return total_loss, recon_loss, diffusion_loss, perceptual


def train_epoch(model, dataloader, optimizer, config, epoch, diffusion_pipe, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_diffusion = 0
    total_perceptual = 0
    num_batches = 0
    
    # Compute diffusion loss less frequently to save memory
    compute_diffusion_every = 5  # Every 5 batches
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for batch_idx, data in enumerate(pbar):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        recon, z = model(data)
        
        # Compute diffusion loss only occasionally to save memory
        use_diffusion = (diffusion_pipe is not None and batch_idx % compute_diffusion_every == 0)
        current_diffusion_pipe = diffusion_pipe if use_diffusion else None
        
        loss, recon_loss, diffusion_loss, perceptual = dgae_loss(
            recon, data, z, model, current_diffusion_pipe, config, device
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_diffusion += diffusion_loss.item() if isinstance(diffusion_loss, torch.Tensor) else diffusion_loss
        total_perceptual += perceptual.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'diff': f'{diffusion_loss.item() if isinstance(diffusion_loss, torch.Tensor) else diffusion_loss:.4f}',
            'perc': f'{perceptual.item():.4f}'
        })
        
        # Clear cache periodically
        if batch_idx % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return (total_loss / num_batches, total_recon / num_batches, 
            total_diffusion / num_batches, total_perceptual / num_batches)


@torch.no_grad()
def generate_samples(model, epoch, output_dir, device, num_samples=16):
    """Generate random samples from DGAE"""
    model.eval()
    z = torch.randn(num_samples, model.latent_dim).to(device)
    samples = model.decode(z)
    
    samples_dir = Path(output_dir) / 'samples'
    samples_dir.mkdir(exist_ok=True, parents=True)
    save_image(samples, samples_dir / f'samples_epoch_{epoch}.png', 
               nrow=4, normalize=True, value_range=(-1, 1))


@torch.no_grad()
def visualize_reconstruction(model, dataloader, epoch, output_dir, device, num_images=8):
    """Visualize reconstructions"""
    model.eval()
    data = next(iter(dataloader))[:num_images].to(device)
    recon, z = model(data)
    
    comparison = torch.cat([data, recon])
    
    recon_dir = Path(output_dir) / 'reconstructions'
    recon_dir.mkdir(exist_ok=True, parents=True)
    save_image(comparison, recon_dir / f'reconstruction_epoch_{epoch}.png',
               nrow=num_images, normalize=True, value_range=(-1, 1))


def plot_losses(losses, output_dir, config):
    """Plot training losses"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(losses['total'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(losses['recon'])
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(losses['diffusion'])
    axes[1, 0].set_title('Diffusion Guidance Loss')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(losses['perceptual'])
    axes[1, 1].set_title('Perceptual Loss')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'training_losses.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train DGAE (Diffusion-Guided Autoencoder) for Corel Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with all classes
  python 4A-train-dgae-corel.py --data-dir training_data/corel/corel_all \\
      --lora-dir corel_models --output-dir dgae_models
  
  # Train for specific class
  python 4A-train-dgae-corel.py --data-dir training_data/corel/class_0001 \\
      --lora-dir corel_models --lora-name lora_...class_0001...safetensors \\
      --output-dir dgae_models/class_0001
  
  # Custom parameters
  python 4A-train-dgae-corel.py --data-dir training_data/corel/corel_all \\
      --lora-dir corel_models --epochs 400 --batch-size 4 --latent-dim 256
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
        '--lora-dir',
        type=str,
        required=True,
        help='Directory containing LoRA weights from Task 2'
    )
    parser.add_argument(
        '--lora-name',
        type=str,
        default=None,
        help='Specific LoRA file name (default: use most recent)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='dgae_models',
        help='Output directory for DGAE model (default: dgae_models)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='.',
        help='Base directory for paths (default: current directory)'
    )
    
    # Model arguments
    parser.add_argument(
        '--pretrained-model',
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        help='Base Stable Diffusion model (default: runwayml/stable-diffusion-v1-5)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=300,
        help='Number of training epochs (default: 300)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Training batch size (default: 8, smaller due to diffusion model memory)'
    )
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=128,
        help='Latent dimension (default: 128, compact for DGAE)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    
    # DGAE-specific arguments
    parser.add_argument(
        '--diffusion-guidance-weight',
        type=float,
        default=0.1,
        help='Weight for diffusion guidance loss (default: 0.1)'
    )
    parser.add_argument(
        '--recon-weight',
        type=float,
        default=1.0,
        help='Weight for reconstruction loss (default: 1.0)'
    )
    
    # Perceptual loss arguments
    parser.add_argument(
        '--no-perceptual',
        action='store_true',
        help='Disable perceptual loss'
    )
    parser.add_argument(
        '--perceptual-weight',
        type=float,
        default=0.03,
        help='Perceptual loss weight (default: 0.03)'
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
    parser.add_argument(
        '--no-diffusion-guidance',
        action='store_true',
        help='Disable diffusion guidance (train regular autoencoder)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir).resolve()
    data_dir = base_dir / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    data_dir = data_dir.resolve()
    lora_dir = base_dir / args.lora_dir if not Path(args.lora_dir).is_absolute() else Path(args.lora_dir)
    lora_dir = lora_dir.resolve()
    output_dir = base_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find LoRA file
    if args.lora_name:
        lora_name = args.lora_name
        lora_file = lora_dir / lora_name
        if not lora_file.exists():
            print(f"ERROR: LoRA file not found: {lora_file}")
            return 1
    else:
        lora_files = list(lora_dir.glob("*.safetensors"))
        if not lora_files:
            print(f"ERROR: No LoRA files found in {lora_dir}")
            return 1
        lora_file = sorted(lora_files, key=lambda x: x.stat().st_mtime)[-1]
        lora_name = lora_file.name
        print(f"Using most recent LoRA: {lora_name}")
    
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
    
    # Update config
    config = Config()
    config.data_dir = str(data_dir)
    config.lora_dir = str(lora_dir)
    config.lora_name = lora_name
    config.pretrained_model = args.pretrained_model
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.latent_dim = args.latent_dim
    config.learning_rate = args.learning_rate
    config.diffusion_guidance_weight = args.diffusion_guidance_weight
    config.recon_weight = args.recon_weight
    config.use_perceptual = not args.no_perceptual
    config.perceptual_weight = args.perceptual_weight
    config.output_dir = str(output_dir)
    config.device = device
    config.num_workers = args.num_workers
    
    print("="*60)
    print("DGAE: DIFFUSION-GUIDED AUTOENCODER TRAINING")
    print("="*60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Base directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"LoRA directory: {lora_dir}")
    print(f"LoRA file: {lora_name}")
    print(f"Output directory: {output_dir}")
    print(f"Latent dimension: {config.latent_dim}")
    print(f"Diffusion guidance: {'ENABLED' if not args.no_diffusion_guidance else 'DISABLED'}")
    print(f"Perceptual loss: {'ENABLED' if config.use_perceptual else 'DISABLED'}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print("="*60 + "\n")
    
    # Load dataset
    dataset = SimpleDataset(config.data_dir, config.image_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if device.startswith("cuda") else False,
        drop_last=True
    )
    
    print(f"✓ Dataset: {len(dataset)} images\n")
    
    # Create model
    model = DGAE(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {num_params:,}\n")
    
    # Load diffusion model for guidance (if enabled)
    diffusion_pipe = None
    if not args.no_diffusion_guidance:
        try:
            diffusion_pipe = load_diffusion_model(
                str(lora_dir), lora_name, args.pretrained_model, device
            )
        except Exception as e:
            print(f"⚠ Warning: Failed to load diffusion model: {e}")
            print("⚠ Continuing without diffusion guidance...")
            diffusion_pipe = None
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Training state
    losses = {'total': [], 'recon': [], 'diffusion': [], 'perceptual': []}
    best_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        best_loss = checkpoint['loss']
        print(f"✓ Resumed from epoch {start_epoch}\n")
    
    print("="*60)
    print("TRAINING...")
    print("="*60 + "\n")
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = time.time()
        
        avg_loss, avg_recon, avg_diffusion, avg_perc = train_epoch(
            model, dataloader, optimizer, config, epoch, diffusion_pipe, device
        )
        
        losses['total'].append(avg_loss)
        losses['recon'].append(avg_recon)
        losses['diffusion'].append(avg_diffusion)
        losses['perceptual'].append(avg_perc)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs}:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Recon Loss: {avg_recon:.4f}")
        print(f"  Diffusion Loss: {avg_diffusion:.4f}")
        print(f"  Perceptual Loss: {avg_perc:.4f}")
        print(f"  Epoch Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses,
                'config': config.__dict__,
            }, Path(output_dir) / 'best_model.pt')
            print(f"  ✓ Saved best model")
        
        # Generate samples and visualizations
        if (epoch + 1) % config.sample_every == 0:
            generate_samples(model, epoch + 1, output_dir, device, 16)
            visualize_reconstruction(model, dataloader, epoch + 1, output_dir, device)
            plot_losses(losses, output_dir, config)
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses,
                'config': config.__dict__,
            }, Path(output_dir) / f'checkpoint_{epoch+1:04d}.pt')
    
    total_time = time.time() - training_start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Total training time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {losses['total'][-1]:.4f}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

