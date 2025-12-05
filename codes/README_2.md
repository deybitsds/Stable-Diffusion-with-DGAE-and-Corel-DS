# Corel Dataset Image Generation with Stable Diffusion

This project adapts Stable Diffusion codes for the Corel dataset:
- **Task 2**: Fine-tuning with LoRA (codes 6 and 7)
- **Task 3**: Training models from scratch (codes 4 and 5)

## Project Structure

```
codes/
├── 2A-prepare_corel_dataset.py         # Task 2: Prepare the dataset
├── 2B-train-lora-corel.py               # Task 2: Train LoRA on Corel
├── 2C-generate-lora-corel.py            # Task 2: Generate images with LoRA
├── 3A-train-vae-corel.py                # Task 3: Train VAE from scratch
├── 3B-train-diffusion-from-scratch-corel.py  # Task 3: Train diffusion model
├── 3C-generate-samples-corel.py        # Task 3: Generate images with diffusion
├── data/
│   └── corel/                           # Original Corel dataset
│       ├── *.png                        # Images (format: XXXX_YYYY.png)
│       └── classes.txt                 # Class mapping (generated if not exists)
├── training_data/
│   └── corel/
│       ├── corel_all/                   # Unified dataset (all classes)
│       │   ├── *.png
│       │   └── captions.json
│       └── class_XXXX/                  # Per-class datasets
│           ├── *.png
│           └── captions.json
├── corel_models/                        # Task 2: Trained LoRA models
│   └── lora_*.safetensors
├── vae_models/                          # Task 3: Trained VAE models
│   ├── best_model.pt
│   ├── checkpoint_*.pt
│   ├── samples/
│   └── reconstructions/
├── diffusion_models/                    # Task 3: Trained diffusion models
│   ├── best_model.pt
│   ├── checkpoint_*.pt
│   └── samples/
└── generated_images/                     # Generated images
    └── corel_*.png
```

## Installation

### Install Dependencies

```bash
# Core dependencies
pip install diffusers accelerate peft transformers datasets torch torchvision

# Optional but recommended for GPU optimization
pip install bitsandbytes  # 8-bit optimizer (saves memory)
pip install xformers      # Memory-efficient attention (optional)

# For data preparation (optional)
pip install pillow matplotlib numpy

# For data augmentation (optional, for VAE training)
pip install albumentations opencv-python
```

## Execution Order

### Task 2: Fine-tuning with LoRA (Pre-trained Model)

#### Step 1: Prepare the Dataset

Generate captions.json and organize the dataset:

```bash
python 2A-prepare_corel_dataset.py --data-dir data/corel --base-dir .
```

**Options:**
- `--data-dir`: Directory containing Corel images (default: `data/corel`)
- `--base-dir`: Base directory for paths (default: `.`)
- `--output-dir`: Output directory for training data (default: `base_dir/training_data`)
- `--figs-dir`: Directory for visualizations (default: `base_dir/figs`)
- `--no-visualize`: Skip dataset visualization

**What it does:**
1. Loads and explores the Corel dataset from `data/corel/`
2. Creates `classes.txt` if it doesn't exist (⚠ **Edit it with actual class names!**)
3. Generates `captions.json` for each class
4. Creates a unified dataset with all classes
5. Visualizes dataset samples

**Output:**
- `training_data/corel/corel_all/` - Unified dataset
- `training_data/corel/class_XXXX/` - Per-class datasets
- `figs/corel_dataset_samples.png` - Visualization

#### Step 2: Train LoRA

#### Option A: Train with All Classes (Unified Model)

```bash
python 2B-train-lora-corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir corel_models \
    --epochs 200 \
    --batch-size 1 \
    --grad-accum 4
```

#### Option B: Train Per Class (Recommended for Few Samples Per Class)

```bash
# Train for class 1
python 2B-train-lora-corel.py \
    --data-dir training_data/corel/class_0001 \
    --output-dir corel_models \
    --epochs 200

# Train for class 2
python 2B-train-lora-corel.py \
    --data-dir training_data/corel/class_0002 \
    --output-dir corel_models \
    --epochs 200

# ... repeat for each class
```

**Training Parameters:**
- `--data-dir`: Dataset directory (must contain `captions.json`) **[REQUIRED]**
- `--output-dir`: Output directory for LoRA weights (default: `corel_models`)
- `--base-dir`: Base directory for paths (default: `.`)
- `--pretrained-model`: Base Stable Diffusion model (default: `runwayml/stable-diffusion-v1-5`)
- `--epochs`: Number of training epochs (default: `200`, **fully configurable**)
  - **Few samples (< 50 images)**: Try 300-500 epochs
  - **Medium dataset (50-200 images)**: 200-300 epochs usually works well
  - **Large dataset (> 200 images)**: 100-200 epochs may be sufficient
  - **Quick testing**: Use 10-50 epochs with `--max-samples 10`
- `--batch-size`: Training batch size (default: `1`)
- `--grad-accum`: Gradient accumulation steps (default: `4`)
- `--learning-rate`: Learning rate (default: `1e-4`)
- `--resolution`: Image resolution (default: `512`)
- `--lora-rank`: LoRA rank (default: `4`)
- `--lora-alpha`: LoRA alpha (default: `4`)
- `--max-samples`: Limit samples for quick testing (default: `None`, use all)
- `--num-workers`: DataLoader workers (default: `4`, set to `0` to disable)
- `--no-8bit-adam`: Disable 8-bit Adam optimizer

**Examples with different epoch counts:**
```bash
# Quick test (10 epochs, limited samples)
python 2B-train-lora-corel.py --data-dir training_data/corel/corel_all --epochs 10 --max-samples 20

# Few samples per class (more epochs needed)
python 2B-train-lora-corel.py --data-dir training_data/corel/class_0001 --epochs 400

# Large dataset (fewer epochs may be enough)
python 2B-train-lora-corel.py --data-dir training_data/corel/corel_all --epochs 100
```

**Output:**
- `corel_models/lora_*.safetensors` - Trained LoRA weights

### Step 3: Generate Images

#### Generate with Most Recent LoRA

```bash
python 2C-generate-lora-corel.py --lora-dir corel_models
```

#### Generate with Specific LoRA File

```bash
python 2C-generate-lora-corel.py \
    --lora-dir corel_models \
    --lora-name lora_stable-diffusion-v1-5_rank4_s800_r512_DDPMScheduler_corel_all_20250101-120000.safetensors
```

#### Custom Prompt and Parameters

```bash
python 2C-generate-lora-corel.py \
    --lora-dir corel_models \
    --prompt "a photo of a royalguard" \
    --num-images 8 \
    --width 768 \
    --height 768 \
    --guidance-scale 7.5 \
    --seed 42
```

**Generation Parameters:**
- `--lora-dir`: Directory with LoRA weights (default: `corel_models`)
- `--lora-name`: Specific LoRA file name (default: use most recent)
- `--output-dir`: Output directory for images (default: `generated_images`)
- `--base-dir`: Base directory for paths (default: `.`)
- `--pretrained-model`: Base Stable Diffusion model (default: `runwayml/stable-diffusion-v1-5`)
- `--prompt`: Text prompt (default: auto-detect from LoRA name)
- `--num-images`: Number of images to generate (default: `4`)
- `--width`, `--height`: Image dimensions (default: `512x512`)
- `--guidance-scale`: Guidance scale (default: `7.5`)
- `--seed`: Random seed (default: `42`)

**Output:**
- `generated_images/corel_*.png` - Individual generated images
- `generated_images/corel_grid_*.png` - Image grids

## Cloud Execution

All scripts support cloud execution via `--base-dir`:

```bash
# Example for cloud GPU rental
python 2A-prepare_corel_dataset.py \
    --data-dir data/corel \
    --base-dir /workspace

python 2B-train-lora-corel.py \
    --data-dir /workspace/training_data/corel/corel_all \
    --output-dir /workspace/corel_models \
    --base-dir /workspace

python 2C-generate-lora-corel.py \
    --lora-dir /workspace/corel_models \
    --output-dir /workspace/generated_images \
    --base-dir /workspace
```

## GPU Optimization

All scripts are optimized for GPU execution:

- **CUDA verification** - Checks GPU availability and displays info
- **CUDA optimizations** - `cudnn.benchmark=True` for faster convolutions
- **Parallel data loading** - DataLoader with `pin_memory` and `num_workers`
- **Memory optimizations** - Attention slicing, VAE tiling, gradient checkpointing
- **8-bit optimizer** - Optional bitsandbytes for memory savings
- **Mixed precision** - FP16 training for faster execution

## When to Use Per-Class vs Unified Model

**Use per-class models if:**
- Few samples per class (< 50)
- Classes are very different
- You need class-specific fine-tuning

**Use unified model if:**
- Many samples per class (> 100)
- Classes are similar
- You want a general model

**Recommendation:** Start with a unified model. If quality is insufficient, train per-class models.

## Troubleshooting

### Error: "No PNG files found"
- Verify that `--data-dir` points to the correct directory
- Check that images follow the format `XXXX_YYYY.png`

### Error: "classes.txt not found"
- Run `2A-prepare_corel_dataset.py` first
- Or create `data/corel/classes.txt` manually with format:
  ```
  1 class_name_1
  2 class_name_2
  ...
  ```

### Error: "CUDA out of memory"
- Reduce `--batch-size` to 1
- Increase `--grad-accum`
- Reduce `--resolution` to 384 or 256
- Use `--max-samples` for quick testing
- Enable `--no-8bit-adam` if using bitsandbytes causes issues

### Error: "No LoRA files found"
- Verify that training completed successfully
- Check that `--output-dir` points to the correct directory
- Ensure LoRA files have `.safetensors` extension

## Output Structure

### Task 2 Outputs (LoRA):
```
corel_models/
└── lora_stable-diffusion-v1-5_rank4_s800_r512_DDPMScheduler_corel_all_YYYYMMDD-HHMMSS.safetensors

generated_images/
├── corel_a_photo_of_a_royalguard_YYYYMMDD_HHMMSS_01.png
├── corel_a_photo_of_a_royalguard_YYYYMMDD_HHMMSS_02.png
├── ...
└── corel_all_grid_YYYYMMDD_HHMMSS.png
```

### Task 3 Outputs (From Scratch):
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

diffusion_models/
├── best_model.pt
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_20.pt
├── ...
├── samples/
│   └── epoch_*.png
└── loss_history.png

generated_images/
└── corel_generated.png (or custom name)
```

### Task 3: Train Models from Scratch

This section covers training VAE and diffusion models from scratch on the Corel dataset.

#### Step 1: Train VAE

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
- `--num-workers`: DataLoader workers (default: `4`)

**Output:**
- `vae_models/best_model.pt` - Best VAE model
- `vae_models/checkpoint_*.pt` - Periodic checkpoints
- `vae_models/samples/` - Generated samples during training
- `vae_models/reconstructions/` - Reconstruction visualizations

#### Step 2: Train Diffusion Model

Train a diffusion model in the latent space of the trained VAE:

```bash
# Train with all classes
python 3B-train-diffusion-from-scratch-corel.py \\
    --vae-checkpoint vae_models/best_model.pt \\
    --image-dir training_data/corel/corel_all \\
    --output-dir diffusion_models

# Train for specific class
python 3B-train-diffusion-from-scratch-corel.py \\
    --vae-checkpoint vae_models/class_0001/best_model.pt \\
    --image-dir training_data/corel/class_0001 \\
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
- `--num-workers`: DataLoader workers (default: `4`)

**Output:**
- `diffusion_models/best_model.pt` - Best diffusion model
- `diffusion_models/checkpoint_*.pt` - Periodic checkpoints
- `diffusion_models/samples/` - Generated samples during training
- `diffusion_models/loss_history.png` - Training loss plot

### Step 3: Generate Images

Generate images using the trained diffusion model:

```bash
# Generate with default settings (DDPM, 1000 steps)
python 3C-generate-samples-corel.py \\
    --diffusion-checkpoint diffusion_models/best_model.pt \\
    --vae-checkpoint vae_models/best_model.pt \\
    --output generated_images.png

# Generate with DDIM (faster, 50 steps)
python 3C-generate-samples-corel.py \\
    --diffusion-checkpoint diffusion_models/best_model.pt \\
    --vae-checkpoint vae_models/best_model.pt \\
    --num-images 32 --ddim-steps 50 --output generated_images.png
```

**Generation Parameters:**
- `--diffusion-checkpoint`: Path to diffusion model checkpoint **[REQUIRED]**
- `--vae-checkpoint`: Path to VAE checkpoint **[REQUIRED]**
- `--output`: Output path for images (default: `generated_images.png`)
- `--base-dir`: Base directory for paths (default: `.`)
- `--num-images`: Number of images to generate (default: `16`)
- `--ddim-steps`: Use DDIM with N steps (faster). If None, uses DDPM (1000 steps)
- `--batch-size`: Batch size for generation (default: `16`)
- `--seed`: Random seed for reproducibility
- `--nrow`: Number of images per row in grid (default: `4`)
- `--device`: Device to use: `cuda` or `cpu` (default: `cuda`)

**Output:**
- `generated_images.png` - Grid of generated images
- `generated_images_individual/` - Individual image files

### When to Use Per-Class vs Unified Models

**Use per-class models if:**
- Few samples per class (< 50)
- Classes are very different
- You need class-specific generation

**Use unified model if:**
- Many samples per class (> 100)
- Classes are similar
- You want a general model

**Recommendation:** Start with a unified model. If quality is insufficient, train per-class models.

### Complete Workflow Examples

#### Task 2 Workflow (LoRA Fine-tuning):

```bash
# 1. Prepare dataset
python 2A-prepare_corel_dataset.py --data-dir data/corel --base-dir .

# 2. Train LoRA
python 2B-train-lora-corel.py --data-dir training_data/corel/corel_all --output-dir corel_models --epochs 200

# 3. Generate images with LoRA
python 2C-generate-lora-corel.py --lora-dir corel_models --prompt "a photo of a corel image" --num-images 8
```

#### Task 3 Workflow (Train from Scratch):

```bash
# 1. Prepare dataset (if not done already)
python 2A-prepare_corel_dataset.py --data-dir data/corel --base-dir .

# 2. Train VAE
python 3A-train-vae-corel.py --data-dir training_data/corel/corel_all --output-dir vae_models --epochs 300

# 3. Train diffusion model
python 3B-train-diffusion-from-scratch-corel.py \\
    --vae-checkpoint vae_models/best_model.pt \\
    --image-dir training_data/corel/corel_all \\
    --output-dir diffusion_models --epochs 500

# 4. Generate images with diffusion model
python 3C-generate-samples-corel.py \\
    --diffusion-checkpoint diffusion_models/best_model.pt \\
    --vae-checkpoint vae_models/best_model.pt \\
    --num-images 16 --output corel_generated.png
```

## References

- Original codes: 
  - Task 2: `code6-train-model-with-lora.py` and `code7-generate-image-with-lora.py`
  - Task 3: `code4-train-vae.py` and `code5-train-stable-diffusion-from-scratch.py`
- Project description: `description.pdf`
