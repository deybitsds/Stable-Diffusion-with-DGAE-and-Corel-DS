#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Images with Corel LoRA

This script generates images using LoRA weights trained on the Corel dataset.

Usage:
    python 2C-generate-lora-corel.py --lora-dir corel_models --prompt "a photo of a royalguard"
"""

from diffusers import StableDiffusionPipeline
import torch
from diffusers.utils import make_image_grid
from diffusers import EulerDiscreteScheduler
from datetime import datetime
from pathlib import Path
import re
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Generate Images with Corel LoRA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with most recent LoRA
  python 2C-generate-lora-corel.py --lora-dir corel_models
  
  # Generate with specific LoRA file
  python 2C-generate-lora-corel.py --lora-dir corel_models \\
      --lora-name lora_stable-diffusion-v1-5_rank4_s800_r512_DDPMScheduler_corel_all_20250101-120000.safetensors
  
  # Custom prompt and parameters
  python 2C-generate-lora-corel.py --lora-dir corel_models \\
      --prompt "a photo of a royalguard" --num-images 8 --width 768 --height 768
        """
    )
    
    # Path arguments
    parser.add_argument(
        '--lora-dir',
        type=str,
        default='corel_models',
        help='Directory containing LoRA weights (default: corel_models)'
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
        default='generated_images',
        help='Output directory for generated images (default: generated_images)'
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
    
    # Generation arguments
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Text prompt (default: auto-detect from LoRA name)'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=4,
        help='Number of images to generate (default: 4)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='Image width (default: 512)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=512,
        help='Image height (default: 512)'
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=7.5,
        help='Guidance scale (default: 7.5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir).resolve()
    lora_dir = base_dir / args.lora_dir if not Path(args.lora_dir).is_absolute() else Path(args.lora_dir)
    lora_dir = lora_dir.resolve()
    output_dir = base_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify CUDA availability and GPU info
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
        
        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("CUDA optimizations enabled (cudnn.benchmark=True)\n")
    else:
        device = "cpu"
        print("\nWARNING: CUDA not available! Generation will be VERY slow on CPU.")
        print("This script is optimized for GPU. Consider using a GPU-enabled environment.")
        print(f"Device: {device}\n")
    
    print("="*60)
    print("GENERATION CONFIGURATION")
    print("="*60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Base directory: {base_dir}")
    print(f"LoRA directory: {lora_dir}")
    print(f"LoRA name: {args.lora_name if args.lora_name else 'Latest'}")
    print(f"Base model: {args.pretrained_model}")
    print(f"Prompt: {args.prompt if args.prompt else 'Auto (from LoRA name)'}")
    print(f"Number of images: {args.num_images}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Seed: {args.seed}")
    print("="*60)
    
    # Find LoRA file
    lora_path = Path(lora_dir)
    
    if args.lora_name:
        lora_file = lora_path / args.lora_name
        if not lora_file.exists():
            print(f"ERROR: LoRA file not found: {lora_file}")
            return 1
        lora_name = args.lora_name
        print(f"\nUsing specific LoRA: {lora_name}")
    else:
        # Find the most recent file
        lora_files = list(lora_path.glob("*.safetensors"))
        if not lora_files:
            print(f"ERROR: No LoRA files found in {lora_dir}")
            return 1
        lora_file = sorted(lora_files, key=lambda x: x.stat().st_mtime)[-1]
        lora_name = lora_file.name
        print(f"\nUsing most recent LoRA: {lora_name}")
    
    print(f"Full file path: {lora_file}")
    
    # Load Stable Diffusion model
    print("\nLoading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.float16,
        safety_checker=None  # Remove safety checker to save VRAM
    ).to(device)
    
    # Enable GPU memory optimizations
    if torch.cuda.is_available():
        pipe.enable_attention_slicing(slice_size=1)  # Reduce memory usage
        pipe.enable_vae_tiling()  # Process VAE in tiles
        pipe.enable_vae_slicing()  # Additional VAE optimization
        print("GPU memory optimizations enabled:")
        print("  - Attention slicing (reduces memory)")
        print("  - VAE tiling (processes in chunks)")
        print("  - VAE slicing (additional optimization)")
    else:
        print("Running on CPU - no GPU optimizations available")
    
    # Load LoRA weights
    print(f"\nLoading LoRA: {lora_name}")
    pipe.load_lora_weights(
        pretrained_model_name_or_path_or_dict=str(lora_path),
        weight_name=lora_name,
        adapter_name="corel_lora"
    )
    pipe.set_adapters(["corel_lora"], adapter_weights=[1.0])
    print("LoRA loaded successfully")
    
    # Configure scheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    print("Scheduler configured")
    
    # Determine prompts based on LoRA name or use provided prompt
    if args.prompt:
        prompts = [args.prompt]
    else:
        # Extract information from LoRA filename
        if "class_" in lora_name:
            # Try to extract class number
            match = re.search(r'class_(\d+)', lora_name)
            if match:
                class_num = match.group(1)
                prompts = [f"a photo of a corel class {class_num}"]
            else:
                prompts = ["a photo of a corel image"]
        elif "corel_all" in lora_name:
            # Generic prompts for all-classes model
            prompts = [
                "a photo of a corel image",
                "a high quality corel image",
                "a detailed corel photograph"
            ]
        else:
            prompts = ["a photo of a corel image"]
    
    print(f"\nPrompts determined: {len(prompts)}")
    for i, p in enumerate(prompts):
        print(f"  {i+1}. {p}")
    
    # Generate images
    all_images = []
    all_prompts = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    negative_prompt = "low quality, blur, watermark, words, name, text"
    
    for prompt_idx, current_prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"GENERATING - Prompt {prompt_idx+1}/{len(prompts)}")
        print(f"{'='*60}")
        print(f"Prompt: {current_prompt}")
        print(f"Resolution: {args.width}x{args.height}")
        print(f"Number of images: {args.num_images}")
        print(f"Guidance scale: {args.guidance_scale}")
        print(f"Seed: {args.seed}")
        print(f"{'='*60}")
        
        # Generate images one at a time for better memory management
        images = []
        
        for i in range(args.num_images):
            print(f"\n[{i+1}/{args.num_images}] Generating image...")
            
            # Create generator on the correct device
            if device.startswith("cuda"):
                generator = torch.Generator(device=device)
            else:
                generator = torch.Generator()
            generator.manual_seed(args.seed + i)
            
            image = pipe(
                prompt=current_prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=1,
                generator=generator,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance_scale
            ).images[0]
            
            images.append(image)
            
            # Save individual image
            safe_prompt = "".join(c for c in current_prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_prompt = safe_prompt.replace(' ', '_')
            img_path = output_dir / f"corel_{safe_prompt}_{timestamp}_{i+1:02d}.png"
            image.save(img_path)
            print(f"  Saved: {img_path.name}")
            
            # Clear cache between generations
            torch.cuda.empty_cache()
        
        all_images.extend(images)
        all_prompts.extend([current_prompt] * len(images))
        
        # Create grid for this prompt
        if len(images) > 1:
            grid = make_image_grid(images, cols=min(4, len(images)), rows=1)
            grid_path = output_dir / f"corel_grid_{safe_prompt}_{timestamp}.png"
            grid.save(grid_path)
            print(f"  Grid saved: {grid_path.name}")
    
    # Create overall grid if multiple prompts
    if len(all_images) > 1:
        print(f"\n{'='*60}")
        print("Creating overall grid...")
        overall_grid = make_image_grid(all_images, cols=min(4, len(all_images)), rows=len(prompts))
        overall_grid_path = output_dir / f"corel_all_grid_{timestamp}.png"
        overall_grid.save(overall_grid_path)
        print(f"Overall grid saved: {overall_grid_path.name}")
    
    print("\n" + "="*60)
    print("GENERATED IMAGES")
    print("="*60)
    print(f"Total images generated: {len(all_images)}")
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    for img_path in sorted(output_dir.glob(f"corel_*_{timestamp}*.png")):
        print(f"  - {img_path.name}")
    print("="*60)
    
    # Cleanup
    pipe.to("cpu")
    torch.cuda.empty_cache()
    print("\nCleanup completed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

