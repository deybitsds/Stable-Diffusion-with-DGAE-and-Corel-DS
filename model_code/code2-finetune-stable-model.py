#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tuning Stable Diffusion - OPTIMIZED FOR 8GB VRAM (RTX 2080)

This script uses several memory optimization techniques:
1. Gradient checkpointing (trades compute for memory)
2. 8-bit AdamW optimizer (50% memory reduction)
3. Lower resolution (384x384 instead of 512x512)
4. XFormers for efficient attention (optional but recommended)

With these optimizations, training should fit in 8GB VRAM.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import os
import argparse

# Try to import bitsandbytes for 8-bit optimizer
try:
    import bitsandbytes as bnb
    USE_8BIT_ADAM = True
    print("✓ Using 8-bit AdamW (saves ~50% memory)")
except ImportError:
    USE_8BIT_ADAM = False
    print("⚠ bitsandbytes not found. Install with: pip install bitsandbytes")
    print("  Using standard AdamW (higher memory usage)")

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR 8GB VRAM
# ============================================================================

class Config:
    # Model
    model_name = "stablediffusionapi/deliberate-v2"
    
    # Training - OPTIMIZED FOR LOW MEMORY
    num_epochs = 10
    batch_size = 1  # Keep at 1 for 8GB
    learning_rate = 1e-5
    gradient_accumulation_steps = 4  # Effective batch size = 4
    
    # MEMORY OPTIMIZATIONS
    resolution = 384  # Lower than 512 to save memory
    enable_xformers = True  # Use efficient attention if available
    gradient_checkpointing = True  # Essential for 8GB
    use_8bit_adam = USE_8BIT_ADAM  # Use 8-bit optimizer if available
    
    # Paths (will be set from command line arguments)
    data_dir = None
    output_dir = "./fine_tuned_model"
    
    # Diffusion
    num_train_timesteps = 1000
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_every = 10

# ============================================================================
# DATASET
# ============================================================================

class SimpleImageTextDataset(Dataset):
    """Simple dataset for image-text pairs."""
    
    def __init__(self, data_dir, tokenizer, resolution=384):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # Load captions
        captions_file = self.data_dir / "captions.json"
        
        if not captions_file.exists():
            print(f"Creating example captions.json at {captions_file}")
            self.data_dir.mkdir(exist_ok=True, parents=True)
            
            image_files = list(self.data_dir.glob("*.jpg")) + \
                         list(self.data_dir.glob("*.png")) + \
                         list(self.data_dir.glob("*.jpeg"))
            
            if len(image_files) == 0:
                raise ValueError(
                    f"No images found in {data_dir}. Please add images!"
                )
            
            captions = {img.name: f"an image of {img.stem}" for img in image_files}
            
            with open(captions_file, 'w') as f:
                json.dump(captions, f, indent=2)
            
            print(f"Created placeholder captions for {len(captions)} images")
        
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)
        
        self.image_files = list(self.captions.keys())
        print(f"Loaded {len(self.image_files)} training examples at {resolution}x{resolution}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.data_dir / img_name
        
        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # Convert to array and normalize to [-1, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = image * 2.0 - 1.0
        
        # Convert to (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        caption = self.captions[img_name]
        
        return {
            "pixel_values": image,
            "caption": caption
        }

# ============================================================================
# MEMORY OPTIMIZATION UTILITIES
# ============================================================================

def enable_memory_optimizations(unet, config):
    """Apply memory optimization techniques."""
    
    # 1. Enable gradient checkpointing (saves ~40% memory)
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("✓ Enabled gradient checkpointing")
    
    # 2. Enable xformers for efficient attention (saves ~30% memory)
    if config.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("✓ Enabled xformers memory efficient attention")
        except Exception as e:
            print(f"⚠ Could not enable xformers: {e}")
            print("  Install with: pip install xformers")
    
    return unet


def get_memory_stats():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def collate_fn(examples):
    """Collate batch."""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    captions = [example["caption"] for example in examples]
    return {"pixel_values": pixel_values, "captions": captions}


def encode_text(captions, tokenizer, text_encoder, device):
    """Encode text to embeddings."""
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )
    
    text_input_ids = text_inputs.input_ids.to(device)
    
    with torch.no_grad():
        encoder_hidden_states = text_encoder(text_input_ids)[0]
    
    return encoder_hidden_states


def training_step(batch, vae, unet, text_encoder, tokenizer, noise_scheduler, config):
    """Single training step."""
    
    # Get batch data
    pixel_values = batch["pixel_values"].to(config.device)
    captions = batch["captions"]
    
    # Convert to float16 to match VAE dtype
    pixel_values = pixel_values.to(dtype=torch.float16)
    
    # Encode images to latent space
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215
    
    # Sample random timesteps
    bsz = latents.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (bsz,),
        device=latents.device
    ).long()
    
    # Add noise
    noise = torch.randn_like(latents)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Get text embeddings
    encoder_hidden_states = encode_text(captions, tokenizer, text_encoder, config.device)
    
    # Predict noise
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    
    # Compute loss
    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
    
    return loss


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Fine-tune Stable Diffusion on custom dataset'
    )
    parser.add_argument(
        'data_folder',
        type=str,
        help='Name of the data folder inside ./training_data/ (e.g., "dogs", "diseases", etc.)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='./training_data',
        help='Base directory containing data folders (default: ./training_data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./fine_tuned_model',
        help='Output directory for the fine-tuned model (default: ./fine_tuned_model)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='Learning rate (default: 1e-5)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size (default: 1, recommended for 8GB VRAM)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=384,
        help='Image resolution (default: 384, use 512 for more VRAM)'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from (e.g., ./fine_tuned_model/checkpoint_epoch_10)'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=1,
        help='Save checkpoint every N epochs (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Create global config instance and set paths
    global config
    config = Config()
    config.data_dir = os.path.join(args.base_dir, args.data_folder)
    config.output_dir = args.output_dir
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.resolution = args.resolution
    
    # Validate training parameters and provide warnings
    if args.epochs > 50 and not args.resume_from:
        print("\n⚠️  WARNING: Training for >50 epochs from scratch may lead to overfitting!")
        print("   Consider using fewer epochs or resuming from a checkpoint.")
    
    if args.learning_rate < 1e-6:
        print("\n⚠️  WARNING: Learning rate is very low (<1e-6).")
        print("   This may result in very slow learning or no improvement.")
    
    if args.learning_rate > 1e-4:
        print("\n⚠️  WARNING: Learning rate is very high (>1e-4).")
        print("   This may lead to unstable training or catastrophic forgetting.")
    
    if args.resume_from and not os.path.exists(args.resume_from):
        print(f"\n❌ ERROR: Checkpoint not found at {args.resume_from}")
        return 1
    
    # Validate data directory exists
    if not os.path.isdir(config.data_dir):
        print(f"Error: Data directory '{config.data_dir}' does not exist!")
        print(f"Please create it and add your training images and captions.json")
        return 1
    
    print("="*80)
    print("Fine-tuning Stable Diffusion (Optimized for RTX 2080 - 8GB VRAM)")
    print("="*80)
    print(f"Data directory: {config.data_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Resume from: {args.resume_from if args.resume_from else 'None (fresh training)'}")
    print(f"Device: {config.device}")
    print(f"Resolution: {config.resolution}x{config.resolution}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Save checkpoints every: {args.save_every} epoch(s)")
    print(f"Gradient checkpointing: {config.gradient_checkpointing}")
    print(f"8-bit Adam: {config.use_8bit_adam}")
    print(f"Output directory: {config.output_dir}\n")
    
    # Check GPU
    if config.device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f}GB\n")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # ========================================================================
    # Load models
    # ========================================================================
    print("Loading models...")
    
    # VAE - frozen
    vae = AutoencoderKL.from_pretrained(
        config.model_name,
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(config.device)
    vae.requires_grad_(False)
    vae.eval()
    
    # Text encoder - frozen
    tokenizer = CLIPTokenizer.from_pretrained(
        config.model_name,
        subfolder="tokenizer"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        config.model_name,
        subfolder="text_encoder",
        torch_dtype=torch.float16
    ).to(config.device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    # UNet - trainable
    if args.resume_from:
        print(f"\nResuming training from checkpoint: {args.resume_from}")
        if not os.path.exists(args.resume_from):
            print(f"ERROR: Checkpoint not found at {args.resume_from}")
            return 1
        
        unet = UNet2DConditionModel.from_pretrained(
            args.resume_from,
            torch_dtype=torch.float16
        ).to(config.device)
        print("Loaded checkpoint successfully!")
    else:
        print(f"\nStarting fresh training from base model: {config.model_name}")
        unet = UNet2DConditionModel.from_pretrained(
            config.model_name,
            subfolder="unet",
            torch_dtype=torch.float16
        ).to(config.device)
    
    # Apply memory optimizations
    unet = enable_memory_optimizations(unet, config)
    unet.train()
    
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.model_name,
        subfolder="scheduler"
    )
    
    print("\nInitial memory usage:")
    get_memory_stats()
    
    # ========================================================================
    # Prepare dataset
    # ========================================================================
    print("\nPreparing dataset...")
    
    dataset = SimpleImageTextDataset(
        config.data_dir,
        tokenizer,
        resolution=config.resolution
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # ========================================================================
    # Setup optimizer (8-bit if available)
    # ========================================================================
    if config.use_8bit_adam:
        print("\nUsing 8-bit AdamW optimizer (saves ~50% memory)")
        optimizer = bnb.optim.AdamW8bit(
            unet.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
    else:
        print("\nUsing standard AdamW optimizer")
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
    
    # ========================================================================
    # Training loop
    # ========================================================================
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print("="*80)
    
    global_step = 0
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Training step
            loss = training_step(
                batch, vae, unet, text_encoder,
                tokenizer, noise_scheduler, config
            )
            
            # Backward pass
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            # Logging
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            
            if global_step % config.log_every == 0:
                avg_loss = epoch_loss / (step + 1)
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Memory stats
        get_memory_stats()
        
        # Save checkpoint based on save_every parameter
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = Path(config.output_dir) / f"checkpoint_epoch_{epoch+1}"
            unet.save_pretrained(checkpoint_path)
            print(f"✓ Saved checkpoint to {checkpoint_path}\n")
        else:
            print(f"Skipping checkpoint save (save_every={args.save_every})\n")
    
    # ========================================================================
    # Save final model
    # ========================================================================
    final_path = Path(config.output_dir) / "final_model"
    unet.save_pretrained(final_path)
    
    print("="*80)
    print(f"Training complete! Model saved to {final_path}")
    print("="*80)
    
    # Save training info
    training_info = {
        "base_model": config.model_name,
        "resolution": config.resolution,
        "num_epochs": config.num_epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "gradient_checkpointing": config.gradient_checkpointing,
        "use_8bit_adam": config.use_8bit_adam,
        "num_training_examples": len(dataset),
        "final_loss": avg_epoch_loss
    }
    
    with open(Path(config.output_dir) / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)


if __name__ == "__main__":
    main()
