# Task 2: Fine-tuning Stable Diffusion with LoRA for Corel Dataset

This guide covers fine-tuning a pre-trained Stable Diffusion model using LoRA (Low-Rank Adaptation) on the Corel dataset.

Based on codes 6 and 7 from the original Stable Diffusion implementation.

## Project Structure

```
codes/
├── 2A-prepare_corel_dataset.py    # Prepare the dataset
├── 2B-train-lora-corel.py          # Train LoRA on Corel
├── 2C-generate-lora-corel.py       # Generate images with LoRA
├── data/
│   └── corel/                      # Original Corel dataset
│       ├── *.png                   # Images (format: XXXX_YYYY.png)
│       └── classes.txt            # Class mapping (generated if not exists)
├── training_data/
│   └── corel/
│       ├── corel_all/              # Unified dataset (all classes)
│       │   ├── *.png
│       │   └── captions.json
│       └── class_XXXX/            # Per-class datasets
│           ├── *.png
│           └── captions.json
├── corel_models/                   # Trained LoRA models
│   └── lora_*.safetensors
└── generated_images/               # Generated images
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

# For data preparation
pip install pillow matplotlib numpy
```

## Execution Order

### Step 1: Prepare the Dataset

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

### Step 2: Train LoRA

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

## Complete Workflow Example

```bash
# 1. Prepare dataset
python 2A-prepare_corel_dataset.py --data-dir data/corel --base-dir .

# 2. Train LoRA
python 2B-train-lora-corel.py --data-dir training_data/corel/corel_all --output-dir corel_models --epochs 200

# 3. Generate images with LoRA
python 2C-generate-lora-corel.py --lora-dir corel_models --prompt "a photo of a corel image" --num-images 8
```

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

### Error: "IndexError: list index out of range" (captions.json)
- Make sure `captions.json` exists in the data directory
- Run `2A-prepare_corel_dataset.py` to generate it
- Check that the file is properly formatted JSON

## Output Structure

After training:
```
corel_models/
└── lora_stable-diffusion-v1-5_rank4_s800_r512_DDPMScheduler_corel_all_YYYYMMDD-HHMMSS.safetensors
```

After generation:
```
generated_images/
├── corel_a_photo_of_a_royalguard_YYYYMMDD_HHMMSS_01.png
├── corel_a_photo_of_a_royalguard_YYYYMMDD_HHMMSS_02.png
├── ...
└── corel_all_grid_YYYYMMDD_HHMMSS.png
```

## References

- Original codes: `code6-train-model-with-lora.py` and `code7-generate-image-with-lora.py`
- Project description: `description.pdf` (Task 2)
