#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train LoRA for Corel Dataset

This script trains a LoRA (Low-Rank Adaptation) model for Stable Diffusion using the Corel dataset.

Training options:
- Unified model: All classes together (corel_all)
- Per-class model: One class at a time (class_XXXX)

Usage:
    python 2B-train-lora-corel.py --data-dir training_data/corel/corel_all --output-dir corel_models
"""

import torch
from accelerate import utils
from accelerate import Accelerator
from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from datasets import load_dataset
from torchvision import transforms
import math
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch.nn.functional as F
from diffusers.utils import convert_state_dict_to_diffusers
import gc
from pathlib import Path
from datetime import datetime
import argparse
import sys
import time
import json


def main():
    parser = argparse.ArgumentParser(
        description='Train LoRA for Corel Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with all classes
  python 2B-train-lora-corel.py --data-dir training_data/corel/corel_all --output-dir corel_models
  
  # Train for specific class
  python 2B-train-lora-corel.py --data-dir training_data/corel/class_0001 --output-dir corel_models
  
  # Custom training parameters
  python 2B-train-lora-corel.py --data-dir training_data/corel/corel_all \\
      --epochs 300 --batch-size 2 --learning-rate 5e-5 --lora-rank 8
        """
    )
    
    # Path arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to dataset directory (must contain captions.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='corel_models',
        help='Output directory for LoRA weights (default: corel_models)'
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
    
    # LoRA arguments
    parser.add_argument(
        '--lora-rank',
        type=int,
        default=4,
        help='LoRA rank (default: 4)'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=4,
        help='LoRA alpha (default: 4)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs (default: 200). Adjust based on dataset size: '
             'few samples (<50) may need 300-500, large datasets (>200) may need 100-200'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Training batch size (default: 1)'
    )
    parser.add_argument(
        '--grad-accum',
        type=int,
        default=4,
        help='Gradient accumulation steps (default: 4)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=512,
        help='Image resolution (default: 512)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit number of samples for quick testing (default: None, use all)'
    )
    
    # Optimization arguments
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of DataLoader workers (default: 4, set to 0 to disable)'
    )
    parser.add_argument(
        '--no-8bit-adam',
        action='store_true',
        help='Disable 8-bit Adam optimizer (use regular AdamW)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir).resolve()
    train_data_dir = base_dir / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    train_data_dir = train_data_dir.resolve()
    output_dir = base_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir = output_dir.resolve()
    
    # Verify CUDA availability
    if not torch.cuda.is_available():
        print("\n" + "="*60)
        print("WARNING: CUDA not available!")
        print("="*60)
        print("This script is optimized for GPU. Training will be VERY slow on CPU.")
        print("Consider using a GPU-enabled environment.")
        response = input("\nContinue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting...")
            return 1
    
    # Setup
    utils.write_basic_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine dataset name from path
    dataset_name = train_data_dir.name
    
    # Configuration
    pretrained_model_name_or_path = args.pretrained_model
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    num_train_epochs = args.epochs
    train_batch_size = args.batch_size
    gradient_accumulation_steps = args.grad_accum
    learning_rate = args.learning_rate
    resolution = args.resolution
    max_samples = args.max_samples
    num_workers = args.num_workers if torch.cuda.is_available() else 0
    use_8bit_adam = not args.no_8bit_adam
    
    formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')
    
    print("="*60)
    print("TRAIN LoRA FOR COREL DATASET")
    print("="*60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Base directory: {base_dir}")
    print(f"Data directory: {train_data_dir}")
    print(f"Dataset name: {dataset_name}")
    print(f"Output directory: {output_dir}")
    print(f"Pretrained model: {pretrained_model_name_or_path}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"Epochs: {num_train_epochs}, Batch size: {train_batch_size}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")
    print("="*60)
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16"
    )
    device = accelerator.device
    
    # Verify CUDA availability and GPU info
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print("CUDA AVAILABLE - GPU OPTIMIZATION ENABLED")
        print(f"{'='*60}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Device: {device}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"{'='*60}\n")
        
        # Set CUDA optimizations for maximum performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("CUDA optimizations enabled:")
        print("  - cudnn.benchmark = True (faster convolutions)")
        print("  - cudnn.deterministic = False (maximum speed)\n")
    
    # Load models
    print("Loading pretrained Stable Diffusion model...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    weight_dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype
    ).to(device)
    tokenizer, text_encoder, vae, unet = (
        pipe.tokenizer, pipe.text_encoder, pipe.vae, pipe.unet
    )
    print("Models loaded on GPU")
    
    # Enable optimizations
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()
    
    # Try to enable xFormers (optional, saves memory)
    try:
        unet.enable_xformers_memory_efficient_attention()
        print("xFormers enabled for UNet")
    except:
        print("xFormers not available")
    
    # Freeze base models
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Move VAE to CPU to save memory
    vae.to('cpu')
    torch.cuda.empty_cache()
    
    # Configure LoRA for UNet
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )
    unet.add_adapter(unet_lora_config)
    
    # Configure LoRA for Text Encoder
    text_encoder_lora_rank = 4
    text_encoder_lora_alpha = 4
    text_encoder_lora_config = LoraConfig(
        r=text_encoder_lora_rank,
        lora_alpha=text_encoder_lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    text_encoder.add_adapter(text_encoder_lora_config)
    print("Text encoder LoRA enabled")
    
    # Convert trainable parameters to fp32
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
    
    for param in text_encoder.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
    
    print("LoRA configured")
    
    # Load dataset
    print(f"\nLoading dataset from {train_data_dir}...")
    
    # Check if captions.json exists
    captions_file = train_data_dir / "captions.json"
    if not captions_file.exists():
        raise FileNotFoundError(
            f"captions.json not found in {train_data_dir}!\n"
            f"Please run 2A-prepare_corel_dataset.py first to generate captions.json"
        )
    
    # Load dataset with explicit caption loading
    try:
        dataset = load_dataset("imagefolder", data_dir=str(train_data_dir))
        train_data = dataset["train"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Limit samples if specified
    if max_samples is not None:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
        print(f"  Using {len(train_data)} samples (limited from {len(dataset['train'])})")
    else:
        print(f"  Using all {len(train_data)} samples")
    
    # Check dataset structure
    dataset_columns = list(train_data.features.keys())
    print(f"  Dataset columns: {dataset_columns}")
    
    # Handle caption column - imagefolder should auto-detect captions.json
    if len(dataset_columns) < 2:
        # Captions not automatically loaded, load manually
        print("  Warning: Captions not automatically detected, loading manually from captions.json...")
        
        with open(captions_file, 'r') as f:
            captions_dict = json.load(f)
        
        # Get list of image files in directory (sorted to match dataset order)
        image_files = sorted(train_data_dir.glob("*.png")) + sorted(train_data_dir.glob("*.jpg")) + sorted(train_data_dir.glob("*.jpeg"))
        
        # Add captions to dataset using index-based matching
        def add_captions(example, idx):
            # Match by index position in sorted file list
            if idx < len(image_files):
                image_file = image_files[idx]
                image_name = image_file.name
                caption = captions_dict.get(image_name, "a photo")
            else:
                caption = "a photo"
            
            return {"text": caption}
        
        train_data = train_data.map(add_captions, with_indices=True)
        dataset_columns = list(train_data.features.keys())
        print(f"  Dataset columns after adding captions: {dataset_columns}")
        print(f"  Loaded {len(captions_dict)} captions from captions.json")
    
    # Get column names
    image_column = dataset_columns[0]
    caption_column = dataset_columns[1] if len(dataset_columns) > 1 else "text"
    
    print(f"  Using image column: '{image_column}'")
    print(f"  Using caption column: '{caption_column}'")
    
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return inputs.input_ids
    
    train_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples
    
    with accelerator.main_process_first():
        train_dataset = train_data.with_transform(preprocess_train)
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    # Optimize DataLoader for GPU (parallel data loading)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    print(f"Dataset prepared: {len(train_dataloader)} batches")
    if torch.cuda.is_available():
        print(f"DataLoader optimized for GPU:")
        print(f"  - num_workers={num_workers} (parallel loading)")
        print(f"  - pin_memory={pin_memory} (faster GPU transfer)")
        print(f"  - persistent_workers={persistent_workers} (efficiency)")
    
    # Configure optimization parameters
    text_encoder_lr = 5e-5
    adam_beta1, adam_beta2 = 0.9, 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08
    lr_scheduler_name = "constant"
    max_grad_norm = 1.0
    
    # Get trainable parameters
    unet_lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    text_encoder_lora_layers = list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
    params_to_optimize = [
        {"params": unet_lora_layers, "lr": learning_rate},
        {"params": text_encoder_lora_layers, "lr": text_encoder_lr}
    ]
    
    # Create optimizer (8-bit if available)
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params_to_optimize,
                betas=(adam_beta1, adam_beta2),
                weight_decay=adam_weight_decay,
                eps=adam_epsilon
            )
            print("Using 8-bit AdamW")
        except ImportError:
            print("bitsandbytes not available, using regular AdamW")
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                betas=(adam_beta1, adam_beta2),
                weight_decay=adam_weight_decay,
                eps=adam_epsilon
            )
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon
        )
    
    lr_scheduler = get_scheduler(lr_scheduler_name, optimizer=optimizer)
    
    # Prepare with accelerator
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )
    
    max_train_steps = num_train_epochs * len(train_dataloader)
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    print(f"Optimizer configured: {max_train_steps} total steps")
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Total epochs: {num_train_epochs}")
    print(f"Batches per epoch: {len(train_dataloader)}")
    print(f"Total steps: {max_train_steps}")
    print(f"Batch size: {train_batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {train_batch_size * gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Text encoder LR: {text_encoder_lr}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"Mixed precision: fp16")
    print(f"Max grad norm: {max_grad_norm}")
    print("="*60 + "\n")
    
    best_loss = float('inf')
    loss_history = []
    epoch_losses = []
    training_start_time = time.time()
    
    for epoch in range(num_train_epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{num_train_epochs}")
        print(f"{'='*60}")
        
        unet.train()
        text_encoder.train()
        
        epoch_loss = 0.0
        num_steps_in_epoch = 0
        train_loss = 0.0
        total_grad_norm = 0.0
        grad_norm_count = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Ensure all tensors are on GPU (explicit for optimization)
                # non_blocking=True allows async transfer for better performance
                pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                
                # Move VAE to GPU for encoding
                vae.to(device)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                vae.to('cpu')
                torch.cuda.empty_cache()
                
                # Sample noise on GPU
                noise = torch.randn_like(latents, device=device)
                
                # Sample timesteps on GPU
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    low=0,
                    high=noise_scheduler.config.num_train_timesteps,
                    size=(batch_size,),
                    device=device,
                    generator=torch.Generator(device=device) if torch.cuda.is_available() else None
                ).long()
                
                # Get text embeddings (input_ids already on GPU)
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Predict
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Accumulate losses
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
                
                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet_lora_layers + text_encoder_lora_layers
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                    total_grad_norm += grad_norm.item()
                    grad_norm_count += 1
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    epoch_loss += train_loss
                    train_loss = 0.0
                    num_steps_in_epoch += 1
                
                current_lr = lr_scheduler.get_last_lr()[0]
                logs = {
                    "epoch": f"{epoch+1}/{num_train_epochs}",
                    "loss": f"{loss.detach().item():.4f}",
                    "avg": f"{avg_loss.item():.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": f"{step+1}/{len(train_dataloader)}"
                }
                if accelerator.sync_gradients and grad_norm_count > 0:
                    logs["grad"] = f"{grad_norm.item():.2f}"
                progress_bar.set_postfix(**logs)
            
            # Periodic cleanup
            if step % 10 == 0 and step > 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Calculate average epoch loss
        if num_steps_in_epoch > 0:
            avg_epoch_loss = epoch_loss / num_steps_in_epoch
            avg_grad_norm = total_grad_norm / grad_norm_count if grad_norm_count > 0 else 0.0
        else:
            avg_epoch_loss = epoch_loss
            avg_grad_norm = 0.0
        
        loss_history.append(avg_epoch_loss)
        epoch_losses.append(avg_epoch_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Check if this is a new best
        is_new_best = avg_epoch_loss < best_loss
        if is_new_best:
            improvement = best_loss - avg_epoch_loss if epoch > 0 else 0
            best_loss = avg_epoch_loss
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.6f}")
        print(f"  Steps in epoch: {num_steps_in_epoch}")
        print(f"  Learning Rate: {current_lr:.2e}")
        if avg_grad_norm > 0:
            print(f"  Average Grad Norm: {avg_grad_norm:.4f}")
        print(f"  Best Loss So Far: {best_loss:.6f}")
        print(f"  Epoch Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
        
        if is_new_best:
            if epoch > 0:
                print(f"  Status: NEW BEST MODEL! Improved by {improvement:.6f}")
            else:
                print(f"  Status: First epoch (baseline)")
        else:
            diff = avg_epoch_loss - best_loss
            print(f"  Status: No improvement (+{diff:.6f} from best)")
        
        # GPU memory info
        if torch.cuda.is_available() and accelerator.is_main_process:
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"  GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        # Progress percentage and time estimates
        progress_pct = ((epoch + 1) / num_train_epochs) * 100
        elapsed_time = time.time() - training_start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = num_train_epochs - (epoch + 1)
        estimated_remaining = avg_time_per_epoch * remaining_epochs
        
        print(f"  Progress: {progress_pct:.1f}% ({epoch+1}/{num_train_epochs} epochs)")
        print(f"  Elapsed: {elapsed_time/60:.1f} min | Estimated remaining: {estimated_remaining/60:.1f} min")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    total_training_time = time.time() - training_start_time
    print(f"Total epochs completed: {len(epoch_losses)}")
    print(f"Total training time: {total_training_time/60:.1f} min ({total_training_time/3600:.2f} hours)")
    print(f"Average time per epoch: {total_training_time/len(epoch_losses)/60:.1f} min")
    print(f"Best loss achieved: {best_loss:.6f}")
    print(f"Final loss: {epoch_losses[-1]:.6f}")
    print(f"Total steps: {max_train_steps}")
    print(f"Loss improvement: {epoch_losses[0] - best_loss:.6f} (from {epoch_losses[0]:.6f} to {best_loss:.6f})")
    print("="*60)
    
    # Save LoRA weights
    print(f"\n{'='*60}")
    print("SAVING LoRA WEIGHTS")
    print(f"{'='*60}")
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Preparing models for saving...")
        unet = unet.to(torch.float32)
        text_encoder = text_encoder.to(torch.float32)
        print("  Models converted to float32")
        
        print("Extracting LoRA weights...")
        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet)
        )
        print(f"  UNet LoRA weights extracted: {len(unet_lora_state_dict)} parameters")
        
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_text_encoder)
        )
        print(f"  Text Encoder LoRA weights extracted: {len(text_encoder_lora_state_dict)} parameters")
        
        weight_name = (
            f"lora_{pretrained_model_name_or_path.split('/')[-1]}_"
            f"rank{lora_rank}_s{max_train_steps}_r{resolution}_"
            f"DDPMScheduler_{dataset_name}_{formatted_date}.safetensors"
        )
        
        print(f"\nSaving LoRA weights to file...")
        print(f"  Filename: {weight_name}")
        print(f"  Directory: {output_dir}")
        
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(output_dir),
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_lora_state_dict,
            safe_serialization=True,
            weight_name=weight_name
        )
        
        # Get file size
        file_path = output_dir / weight_name
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        print(f"\n{'='*60}")
        print("LoRA SAVED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"File: {weight_name}")
        print(f"Location: {output_dir}")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Includes: UNet LoRA + Text Encoder LoRA")
        print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
        print(f"Training steps: {max_train_steps}")
        print(f"Best loss: {best_loss:.6f}")
        print(f"{'='*60}")
    else:
        print("Waiting for main process to save weights...")
    
    accelerator.end_training()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

