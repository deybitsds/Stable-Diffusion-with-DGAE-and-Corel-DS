#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chapter 05: Understand How Stable Diffusion Works
Fixed version without Jupyter dependencies
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.utils import load_image
from transformers import CLIPTokenizer, CLIPTextModel

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess(image):
    """Convert PIL image to preprocessed numpy array for VAE encoding."""
    # Convert image from 0~255 to 0~1
    image_array = np.array(image).astype(np.float16) / 255.0
    
    # Convert from 0~1 to -1~1
    image_array = image_array * 2.0 - 1.0
    
    # Transform from (height, width, channel) to (channel, height, width)
    image_array_cwh = image_array.transpose(2, 0, 1)
    
    # Add batch dimension
    image_array_cwh = np.expand_dims(image_array_cwh, axis=0)
    return image_array_cwh


def image_to_latent(image_array_cwh, vae_model):
    """Encode image to latent space using VAE."""
    # Convert to torch tensor
    image_tensor = torch.from_numpy(image_array_cwh).to("cuda:0")
    
    # Encode to latent
    with torch.no_grad():
        latents = vae_model.encode(image_tensor).latent_dist.sample()
    return latents


def latent_to_img(latents_input, vae_model):
    """Decode latent representation back to image."""
    # Decode image
    with torch.no_grad():
        decode_image = vae_model.decode(latents_input, return_dict=False)[0][0]
    
    # Normalize to [0, 1]
    decode_image = (decode_image / 2 + 0.5).clamp(0, 1)
    
    # Move to CPU
    decode_image = decode_image.to("cpu")
    
    # Convert to numpy array
    numpy_img = decode_image.detach().numpy()
    
    # Transform from (channel, height, width) to (height, width, channel)
    numpy_img_t = numpy_img.transpose(1, 2, 0)
    
    # Convert to uint8
    numpy_img_uint8 = (numpy_img_t * 255).round().astype("uint8")
    
    return Image.fromarray(numpy_img_uint8)


# ============================================================================
# LOAD AND INITIALIZE MODELS
# ============================================================================

print("Loading models...")

# Load VAE
vae_model = AutoencoderKL.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    subfolder="vae",
    torch_dtype=torch.float16
).to("cuda:0")

# Load CLIP tokenizer and text encoder
clip_tokenizer = CLIPTokenizer.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    subfolder="tokenizer"
)

clip_text_encoder = CLIPTextModel.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    subfolder="text_encoder",
    torch_dtype=torch.float16
).to("cuda")

# Load UNet
unet = UNet2DConditionModel.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    subfolder="unet",
    torch_dtype=torch.float16
).to("cuda")

# Load scheduler
scheduler = EulerDiscreteScheduler.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    subfolder="scheduler"
)

print("Models loaded successfully!\n")

# ============================================================================
# LOAD AND PROCESS INPUT IMAGE
# ============================================================================

print("Loading input image...")
image = load_image("./figs/dog.png")
w, h = image.size
print(f"Image dimensions: {w}x{h}")

# Preprocess image
image_array_cwh = preprocess(image)
print(f"Preprocessed array shape: {image_array_cwh.shape}")

# Encode to latent space
latents_input = image_to_latent(image_array_cwh, vae_model)
print(f"Latent shape: {latents_input.shape}\n")

# ============================================================================
# PREPARE TEXT EMBEDDINGS
# ============================================================================

print("Preparing text embeddings...")

input_prompt = "a running dog"
negative_prompt = "blur"

# Tokenize and encode positive prompt
input_tokens = clip_tokenizer(input_prompt, return_tensors="pt")["input_ids"]

with torch.no_grad():
    prompt_embeds = clip_text_encoder(input_tokens.to("cuda"))[0]

# Tokenize and encode negative prompt
max_length = prompt_embeds.shape[1]
uncond_input_tokens = clip_tokenizer(
    negative_prompt,
    padding="max_length",
    max_length=max_length,
    truncation=True,
    return_tensors="pt"
)["input_ids"]

with torch.no_grad():
    negative_prompt_embeds = clip_text_encoder(uncond_input_tokens.to("cuda"))[0]

# Concatenate for classifier-free guidance
prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
print(f"Prompt embeddings shape: {prompt_embeds.shape}\n")

# ============================================================================
# SETUP INFERENCE PARAMETERS
# ============================================================================

inference_steps = 20
scheduler.set_timesteps(inference_steps, device="cuda")
timesteps = scheduler.timesteps

print(f"Number of inference steps: {len(timesteps)}")
print(f"Timesteps: {timesteps.tolist()}\n")

# ============================================================================
# TEXT-TO-IMAGE GENERATION
# ============================================================================

print("Running text-to-image generation...")

# Prepare initial noise
shape = torch.Size([1, 4, 64, 64])
noise_tensor = torch.randn(shape, dtype=torch.float16).to("cuda")
latents_t2i = noise_tensor * scheduler.init_noise_sigma

guidance_scale = 7.5

# Denoising loop
for i, t in enumerate(timesteps):
    # Expand latents for classifier-free guidance
    latent_model_input = torch.cat([latents_t2i] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    
    # Predict noise
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False
        )[0]
    
    # Perform classifier-free guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # Compute previous step
    latents_t2i = scheduler.step(noise_pred, t, latents_t2i, return_dict=False)[0]

# Decode to image
latents_scaled = (1 / 0.18215) * latents_t2i
img_t2i = latent_to_img(latents_scaled, vae_model)
print("Text-to-image generation complete!\n")

# ============================================================================
# IMAGE-TO-IMAGE GENERATION
# ============================================================================

print("Running image-to-image generation...")

# Prepare latents with input image + noise
strength = 0.3
noise_tensor = torch.randn(shape, dtype=torch.float16).to("cuda")
latents_i2i = latents_input * strength + noise_tensor * scheduler.init_noise_sigma

# Reset scheduler
scheduler.set_timesteps(inference_steps, device="cuda")
timesteps = scheduler.timesteps

# Denoising loop
for i, t in enumerate(timesteps):
    # Expand latents for classifier-free guidance
    latent_model_input = torch.cat([latents_i2i] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    
    # Predict noise
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False
        )[0]
    
    # Perform classifier-free guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # Compute previous step
    latents_i2i = scheduler.step(noise_pred, t, latents_i2i, return_dict=False)[0]

# Decode to image
latents_scaled = (1 / 0.18215) * latents_i2i
img_i2i = latent_to_img(latents_scaled, vae_model)
print("Image-to-image generation complete!\n")

# ============================================================================
# VISUALIZE RESULTS
# ============================================================================

print("Displaying results...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(img_t2i)
axes[1].set_title(f'Text-to-Image\n"{input_prompt}"')
axes[1].axis('off')

axes[2].imshow(img_i2i)
axes[2].set_title(f'Image-to-Image\n"{input_prompt}" (strength={strength})')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('./figs/stable_diffusion_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nDone! Results saved to 'stable_diffusion_results.png'")
