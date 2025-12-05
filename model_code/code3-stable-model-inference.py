#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference with Fine-tuned Stable Diffusion Model

This script demonstrates the complete inference process:
1. Load the fine-tuned UNet
2. Encode text prompt with CLIP
3. Start from random noise
4. Iteratively denoise using the UNet (reverse diffusion)
5. Decode latents to image with VAE
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from pathlib import Path
from datetime import datetime
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model paths
    base_model = "stablediffusionapi/deliberate-v2"
    finetuned_unet_path = None  # Will be set from command-line arguments
    
    # Generation parameters
    negative_prompt = "blurry, low quality, distorted, ugly"
    num_inference_steps = 30
    guidance_scale = 7.5
    
    # Image settings
    height = 512
    width = 512
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output
    output_dir = "./figs"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def latent_to_image(latents, vae):
    """
    Decode latent representation to image.
    
    This is the final step: VAE decoder converts latents back to pixel space.
    """
    # Scale latents
    latents = 1 / 0.18215 * latents
    
    with torch.no_grad():
        image = vae.decode(latents).sample
    
    # Normalize from [-1, 1] to [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    
    # Convert to PIL Image
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(img) for img in image]
    
    return pil_images


def prepare_text_embeddings(prompt, negative_prompt, tokenizer, text_encoder, device):
    """
    Prepare text embeddings for classifier-free guidance.
    
    Returns both conditional (prompt) and unconditional (negative) embeddings.
    """
    # Encode positive prompt
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
    
    # Encode negative prompt
    uncond_inputs = tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device))[0]
    
    # Concatenate for classifier-free guidance
    # [unconditional, conditional]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    return text_embeddings


def generate_image(
    prompt,
    negative_prompt,
    unet,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    config,
    seed=None
):
    """
    Generate an image using the reverse diffusion process.
    
    This is the core inference loop:
    1. Start with random noise
    2. For each timestep, predict and remove noise
    3. Decode final latents to image
    """
    
    # Set random seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device=config.device).manual_seed(seed)
    else:
        generator = None
    
    # ========================================================================
    # STEP 1: Prepare text embeddings
    # ========================================================================
    print(f"Encoding prompt: '{prompt}'")
    text_embeddings = prepare_text_embeddings(
        prompt, negative_prompt, tokenizer, text_encoder, config.device
    )
    
    # ========================================================================
    # STEP 2: Prepare random latents (pure noise)
    # ========================================================================
    # Latent shape: (batch, channels, height//8, width//8)
    latents = torch.randn(
        (1, 4, config.height // 8, config.width // 8),
        generator=generator,
        device=config.device,
        dtype=torch.float16
    )
    
    # Scale initial noise by scheduler's init_noise_sigma
    latents = latents * scheduler.init_noise_sigma
    
    # ========================================================================
    # STEP 3: Denoising loop (reverse diffusion)
    # ========================================================================
    print(f"Running reverse diffusion for {config.num_inference_steps} steps...")
    
    scheduler.set_timesteps(config.num_inference_steps, device=config.device)
    
    for i, t in enumerate(scheduler.timesteps):
        # Expand latents for classifier-free guidance
        # We process both conditional and unconditional in one forward pass
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise residual
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample
        
        # Perform classifier-free guidance
        # This combines unconditional and conditional predictions
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        
        # Compute previous noisy sample: x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # Progress indicator
        if (i + 1) % 5 == 0 or (i + 1) == config.num_inference_steps:
            print(f"  Step {i+1}/{config.num_inference_steps}")
    
    # ========================================================================
    # STEP 4: Decode latents to image
    # ========================================================================
    print("Decoding latents to image...")
    images = latent_to_image(latents, vae)
    
    return images[0]


def get_user_prompt():
    """
    Get text prompt from user input.
    """
    print("\n" + "="*80)
    print("IMAGE GENERATION WITH STABLE DIFFUSION")
    print("="*80)
    print("\nEnter your prompt to generate an image.")
    print("Examples:")
    print("  - a happy dog with sunglasses wearing a bear hat and jumping on grass")
    print("  - a cat sitting on a table, oil painting style")
    print("  - a futuristic city at sunset, cyberpunk style")
    print("  - a beautiful landscape with mountains and a lake")
    print("\n" + "-"*80)
    
    prompt = input("\nYour prompt: ").strip()
    
    if not prompt:
        print("\nNo prompt entered. Using default prompt...")
        prompt = "a happy dog with sunglasses wearing a bear hat and jumping on grass"
    
    print(f"\nPrompt accepted: '{prompt}'")
    
    return prompt


# ============================================================================
# MAIN INFERENCE
# ============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate images with fine-tuned Stable Diffusion model'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to fine-tuned UNet model (e.g., ./fine_tuned_model/final_model)'
    )
    parser.add_argument(
        'prompt',
        type=str,
        help='Text prompt for image generation (e.g., "a happy dog with sunglasses")'
    )
    parser.add_argument(
        '--negative-prompt',
        type=str,
        default='blurry, low quality, distorted, ugly',
        help='Negative prompt (default: "blurry, low quality, distorted, ugly")'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=30,
        help='Number of inference steps (default: 30)'
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=7.5,
        help='Guidance scale for classifier-free guidance (default: 7.5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./figs',
        help='Output directory for generated images (default: ./figs)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=512,
        help='Image height (default: 512)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='Image width (default: 512)'
    )
    
    args = parser.parse_args()
    
    # Create global config instance and set parameters
    global config
    config = Config()
    config.finetuned_unet_path = args.model_path
    config.negative_prompt = args.negative_prompt
    config.num_inference_steps = args.steps
    config.guidance_scale = args.guidance_scale
    config.output_dir = args.output_dir
    config.height = args.height
    config.width = args.width
    
    print("="*80)
    print("Inference with Fine-tuned Stable Diffusion")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Fine-tuned UNet: {config.finetuned_unet_path}")
    
    # Create output directory
    Path(config.output_dir).mkdir(exist_ok=True, parents=True)
    
    # ========================================================================
    # STEP 1: Load models
    # ========================================================================
    print("\nLoading models...")
    
    # Load VAE (frozen, for decoding only)
    vae = AutoencoderKL.from_pretrained(
        config.base_model,
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(config.device)
    vae.eval()
    
    # Load text encoder (frozen, for encoding prompts)
    tokenizer = CLIPTokenizer.from_pretrained(
        config.base_model,
        subfolder="tokenizer"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        config.base_model,
        subfolder="text_encoder",
        torch_dtype=torch.float16
    ).to(config.device)
    text_encoder.eval()
    
    # Load fine-tuned UNet
    unet_path = Path(config.finetuned_unet_path)
    
    if not unet_path.exists():
        print(f"\nWARNING: Fine-tuned model not found at {unet_path}")
        print("Using base model UNet instead...")
        print("To use a fine-tuned model, first run 'finetune_stable_diffusion.py'\n")
        
        unet = UNet2DConditionModel.from_pretrained(
            config.base_model,
            subfolder="unet",
            torch_dtype=torch.float16
        ).to(config.device)
    else:
        print(f"Loading fine-tuned UNet from {unet_path}")
        unet = UNet2DConditionModel.from_pretrained(
            unet_path,
            torch_dtype=torch.float16
        ).to(config.device)
    
    unet.eval()
    
    # Load scheduler
    scheduler = EulerDiscreteScheduler.from_pretrained(
        config.base_model,
        subfolder="scheduler"
    )
    
    print("Models loaded successfully!")
    
    # ========================================================================
    # STEP 2: Generate image
    # ========================================================================
    user_prompt = args.prompt
    
    print("\n" + "="*80)
    print("Generating image...")
    print(f"Prompt: '{user_prompt}'")
    print(f"Negative prompt: '{config.negative_prompt}'")
    print(f"Guidance scale: {config.guidance_scale}")
    print(f"Inference steps: {config.num_inference_steps}")
    print("="*80 + "\n")
    
    image = generate_image(
        user_prompt,
        config.negative_prompt,
        unet,
        vae,
        text_encoder,
        tokenizer,
        scheduler,
        config,
        seed=args.seed  # Use seed from command-line arguments
    )
    
    # ========================================================================
    # STEP 3: Save image
    # ========================================================================
    # Create filename from timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"generated_{timestamp}.png"
    output_path = Path(config.output_dir) / output_filename
    
    image.save(output_path)
    print(f"\n{'='*80}")
    print(f"Image saved to: {output_path}")
    print("="*80)
    
    # ========================================================================
    # STEP 4: Display image
    # ========================================================================
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f'Generated Image\n"{user_prompt}"', fontsize=12, wrap=True)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()
