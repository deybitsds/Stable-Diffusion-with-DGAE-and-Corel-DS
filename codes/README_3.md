# Task 3: Train Models from Scratch for Corel Dataset

This guide covers training VAE and diffusion models from scratch on the Corel dataset.

Based on codes 4 and 5 from the original Stable Diffusion implementation.

## Project Structure

```
codes/
├── 3A-train-vae-corel.py                # Train VAE from scratch
├── 3B-train-diffusion-from-scratch-corel.py  # Train diffusion model
├── 3C-generate-samples-corel.py         # Generate images with diffusion
├── training_data/
│   └── corel/
│       ├── corel_all/                   # Unified dataset (all classes)
│       │   ├── *.png
│       │   └── captions.json
│       └── class_XXXX/                  # Per-class datasets
│           ├── *.png
│           └── captions.json
├── vae_models/                          # Trained VAE models
│   ├── best_model.pt
│   ├── checkpoint_*.pt
│   ├── samples/
│   └── reconstructions/
├── diffusion_models/                    # Trained diffusion models
│   ├── best_model.pt
│   ├── checkpoint_*.pt
│   └── samples/
└── generated_images/                    # Generated images
    └── *.png
```

## Installation

### Install Dependencies

```bash
# Core dependencies
pip install torch torchvision matplotlib numpy pillow tqdm

# For data augmentation (optional, for VAE training)
pip install albumentations opencv-python
```

**Note:** The dataset should be prepared first using `2A-prepare_corel_dataset.py` from Task 2.

## Execution Order

### Step 1: Train VAE

Train a Variational Autoencoder (VAE) to learn latent representations:

```bash
# Train with all classes
python 3A-train-vae-corel.py --data-dir training_data/corel/corel_all --output-dir vae_models

# Train for specific class
python 3A-train-vae-corel.py --data-dir training_data/corel/class_0001 --output-dir vae_models/class_0001
```

**VAE Training Parameters:**
- `--data-dir`: Directory containing training images **[REQUIRED]**
- `--output-dir`: Output directory for VAE models (default: `vae_models`)
- `--base-dir`: Base directory for paths (default: `.`)
- `--epochs`: Number of training epochs (default: `300`)
- `--batch-size`: Training batch size (default: `16`)
- `--latent-dim`: Latent dimension (default: `128`)
- `--learning-rate`: Learning rate (default: `1e-4`)
- `--kl-weight`: Final KL weight (default: `1.0`)
- `--kl-warmup`: KL warmup epochs (default: `150`)
- `--kl-target`: KL target value (default: `25.0`)
- `--no-perceptual`: Disable perceptual loss
- `--perceptual-weight`: Perceptual loss weight (default: `0.03`)
- `--resume`: Resume from checkpoint path
- `--seed`: Random seed (default: `42`)
- `--num-workers`: DataLoader workers (default: `4`, set to `0` to disable)

**Output:**
- `vae_models/best_model.pt` - Best VAE model
- `vae_models/checkpoint_*.pt` - Periodic checkpoints (every 20 epochs)
- `vae_models/samples/` - Generated samples during training (every 10 epochs)
- `vae_models/reconstructions/` - Reconstruction visualizations (every 10 epochs)
- `vae_models/training_losses.png` - Training loss plots

**Example with custom parameters:**
```bash
# Train with more epochs and larger latent dimension
python 3A-train-vae-corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir vae_models \
    --epochs 400 \
    --latent-dim 256 \
    --batch-size 32
```

### Step 2: Train Diffusion Model

Train a diffusion model in the latent space of the trained VAE:

```bash
# Train with all classes
python 3B-train-diffusion-from-scratch-corel.py \
    --vae-checkpoint vae_models/best_model.pt \
    --image-dir training_data/corel/corel_all \
    --output-dir diffusion_models

# Train for specific class
python 3B-train-diffusion-from-scratch-corel.py \
    --vae-checkpoint vae_models/class_0001/best_model.pt \
    --image-dir training_data/corel/class_0001 \
    --output-dir diffusion_models/class_0001
```

**Diffusion Training Parameters:**
- `--vae-checkpoint`: Path to VAE checkpoint from 3A **[REQUIRED]**
- `--image-dir`: Directory containing training images **[REQUIRED]**
- `--output-dir`: Output directory (default: `diffusion_models`)
- `--base-dir`: Base directory for paths (default: `.`)
- `--epochs`: Number of training epochs (default: `500`)
- `--batch-size`: Training batch size (default: `16`)
- `--grad-accum`: Gradient accumulation steps (default: `1`)
- `--lr`: Learning rate (default: `1e-4`)
- `--image-size`: Image size (default: `128`)
- `--no-ema`: Disable EMA (Exponential Moving Average)
- `--ema-decay`: EMA decay rate (default: `0.995`)
- `--schedule`: Diffusion schedule type: `linear` or `cosine` (default: `linear`)
- `--no-mixed-precision`: Disable mixed precision training (FP16)
- `--lr-scheduler`: LR scheduler: `cosine`, `cosine_warm_restarts`, `plateau`, `none` (default: `cosine_warm_restarts`)
- `--lr-min`: Minimum learning rate (default: `1e-5`)
- `--early-stopping-patience`: Early stopping patience (default: `30`)
- `--early-stopping-delta`: Early stopping minimum delta (default: `1e-5`)
- `--oscillation-window`: Oscillation detection window size (default: `10`)
- `--oscillation-threshold`: Oscillation detection threshold (default: `0.0005`)
- `--num-workers`: DataLoader workers (default: `4`, set to `0` to disable)
- `--resume`: Resume from checkpoint path

**Output:**
- `diffusion_models/best_model.pt` - Best diffusion model
- `diffusion_models/checkpoint_epoch_*.pt` - Periodic checkpoints (every 10 epochs)
- `diffusion_models/samples/` - Generated samples during training (every 10 epochs)
- `diffusion_models/loss_history.png` - Training loss plot

**Example with custom parameters:**
```bash
# Train with more epochs and gradient accumulation
python 3B-train-diffusion-from-scratch-corel.py \
    --vae-checkpoint vae_models/best_model.pt \
    --image-dir training_data/corel/corel_all \
    --output-dir diffusion_models \
    --epochs 600 \
    --batch-size 8 \
    --grad-accum 2
```

### Step 3: Generate Images

Generate images using the trained diffusion model:

```bash
# Generate with default settings (DDPM, 1000 steps)
python 3C-generate-samples-corel.py \
    --diffusion-checkpoint diffusion_models/best_model.pt \
    --vae-checkpoint vae_models/best_model.pt \
    --output generated_images.png

# Generate with DDIM (faster, 50 steps)
python 3C-generate-samples-corel.py \
    --diffusion-checkpoint diffusion_models/best_model.pt \
    --vae-checkpoint vae_models/best_model.pt \
    --num-images 32 --ddim-steps 50 --output generated_images.png
```

**Generation Parameters:**
- `--diffusion-checkpoint`: Path to diffusion model checkpoint **[REQUIRED]**
- `--vae-checkpoint`: Path to VAE checkpoint **[REQUIRED]**
- `--output`: Output path for images (default: `generated_images.png`)
- `--base-dir`: Base directory for paths (default: `.`)
- `--num-images`: Number of images to generate (default: `16`)
- `--ddim-steps`: Use DDIM with N steps (faster). If None, uses DDPM (1000 steps)
- `--batch-size`: Batch size for generation (default: `16`, reduce if OOM)
- `--seed`: Random seed for reproducibility
- `--nrow`: Number of images per row in grid (default: `4`)
- `--device`: Device to use: `cuda` or `cpu` (default: `cuda`)

**Output:**
- `generated_images.png` - Grid of generated images
- `generated_images_individual/` - Individual image files (if <= 16 images)

**Example with custom parameters:**
```bash
# Generate many images with DDIM (faster)
python 3C-generate-samples-corel.py \
    --diffusion-checkpoint diffusion_models/best_model.pt \
    --vae-checkpoint vae_models/best_model.pt \
    --num-images 64 \
    --ddim-steps 50 \
    --output corel_64_images.png \
    --nrow 8
```

## Cloud Execution

All scripts support cloud execution via `--base-dir`:

```bash
# Example for cloud GPU rental
python 3A-train-vae-corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir /workspace/vae_models \
    --base-dir /workspace

python 3B-train-diffusion-from-scratch-corel.py \
    --vae-checkpoint /workspace/vae_models/best_model.pt \
    --image-dir /workspace/training_data/corel/corel_all \
    --output-dir /workspace/diffusion_models \
    --base-dir /workspace

python 3C-generate-samples-corel.py \
    --diffusion-checkpoint /workspace/diffusion_models/best_model.pt \
    --vae-checkpoint /workspace/vae_models/best_model.pt \
    --output /workspace/generated_images.png \
    --base-dir /workspace
```

## GPU Optimization

All scripts are optimized for GPU execution:

- **CUDA verification** - Checks GPU availability and displays info
- **CUDA optimizations** - `cudnn.benchmark=True` for faster convolutions
- **Parallel data loading** - DataLoader with `pin_memory` and `num_workers`
- **Mixed precision** - FP16 training for faster execution (diffusion model)
- **Memory optimizations** - Efficient batch processing and gradient accumulation

## When to Use Per-Class vs Unified Models

**Use per-class models if:**
- Few samples per class (< 50)
- Classes are very different
- You need class-specific generation

**Use unified model if:**
- Many samples per class (> 100)
- Classes are similar
- You want a general model

**Recommendation:** Start with a unified model. If quality is insufficient, train per-class models.

## Complete Workflow Example

```bash
# 1. Prepare dataset (if not done already - from Task 2)
python 2A-prepare_corel_dataset.py --data-dir data/corel --base-dir .

# 2. Train VAE
python 3A-train-vae-corel.py --data-dir training_data/corel/corel_all --output-dir vae_models --epochs 300

# 3. Train diffusion model
python 3B-train-diffusion-from-scratch-corel.py \
    --vae-checkpoint vae_models/best_model.pt \
    --image-dir training_data/corel/corel_all \
    --output-dir diffusion_models --epochs 500

# 4. Generate images with diffusion model
python 3C-generate-samples-corel.py \
    --diffusion-checkpoint diffusion_models/best_model.pt \
    --vae-checkpoint vae_models/best_model.pt \
    --num-images 16 --output corel_generated.png
```

## Troubleshooting

### Error: "No images found"
- Verify that `--data-dir` points to the correct directory
- Check that images are in PNG, JPG, or JPEG format
- Ensure the directory contains image files

### Error: "VAE checkpoint not found"
- Make sure you've trained the VAE first using `3A-train-vae-corel.py`
- Verify the checkpoint path is correct
- Check that the checkpoint file exists and is readable

### Error: "CUDA out of memory"
- Reduce `--batch-size` (e.g., from 16 to 8 or 4)
- Increase `--grad-accum` to maintain effective batch size
- Reduce `--image-size` if training VAE (e.g., from 128 to 64)
- Use `--num-workers 0` to reduce memory overhead

### Error: "Loss not decreasing"
- Increase number of epochs
- Adjust learning rate (try `--lr 2e-4` or `--lr 5e-5`)
- Check that dataset is properly prepared
- Verify VAE training completed successfully before training diffusion model

### Error: "KL divergence too high/low"
- Adjust `--kl-weight` (increase if too low, decrease if too high)
- Adjust `--kl-warmup` (more warmup epochs for stability)
- Check `--kl-target` value (default 25.0)

## Output Structure

After VAE training:
```
vae_models/
├── best_model.pt
├── checkpoint_0020.pt
├── checkpoint_0040.pt
├── ...
├── samples/
│   └── samples_epoch_*.png
├── reconstructions/
│   └── reconstruction_epoch_*.png
└── training_losses.png
```

After diffusion training:
```
diffusion_models/
├── best_model.pt
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_20.pt
├── ...
├── samples/
│   └── epoch_*.png
└── loss_history.png
```

After generation:
```
generated_images/
├── generated_images.png (or custom name)
└── generated_images_individual/  (if <= 16 images)
    └── image_*.png
```

## References

- Original codes: `code4-train-vae.py` and `code5-train-stable-diffusion-from-scratch.py`
- Project description: `description.pdf` (Task 3)

