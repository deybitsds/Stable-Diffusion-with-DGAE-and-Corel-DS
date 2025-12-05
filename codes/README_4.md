# Task 4: DGAE (Diffusion-Guided Autoencoder) for Latent Representation Learning

This guide covers implementing DGAE (Diffusion-Guided Autoencoder) for learning latent representations from the Corel dataset. The goal is to learn features for clustering evaluation, **NOT for data augmentation**.

Based on the DGAE paper: [DGAE: Diffusion-Guided Autoencoder for Efficient Latent Representation Learning](https://arxiv.org/abs/2506.09644)

## Overview

DGAE uses a pre-trained Stable Diffusion model (with LoRA weights from Task 2) to guide the decoder of a convolutional autoencoder. This approach:

- **Learns compact latent representations** (2x smaller than typical VAE)
- **Improves reconstruction quality** under high compression rates
- **Uses diffusion model as guidance** to help decoder recover informative signals
- **Focuses on clustering**, not data augmentation

## Project Structure

```
codes/
├── 4A-train-dgae-corel.py      # Train DGAE autoencoder
├── 4B-extract-features-corel.py # Extract latent features for clustering
├── corel_models/                # LoRA weights from Task 2 (required)
│   └── lora_*.safetensors
├── dgae_models/                 # Trained DGAE models
│   └── best_model.pt
└── features/                    # Extracted features for clustering
    ├── dgae_features.npy
    ├── dgae_features.json
    └── dgae_features.labels.npy
```

## Installation

### Install Dependencies

```bash
# Core dependencies
pip install torch torchvision diffusers accelerate transformers

# For feature extraction and clustering
pip install numpy scikit-learn matplotlib

# Optional: for advanced clustering evaluation
pip install scipy seaborn
```

## Execution Order

### Prerequisites

**IMPORTANT:** You must complete Task 2 first to obtain LoRA weights:

```bash
# Train LoRA (Task 2)
python 2B-train-lora-corel.py --data-dir training_data/corel/corel_all --output-dir corel_models
```

### Step 1: Train DGAE Autoencoder

Train a DGAE model using the pre-trained LoRA weights from Task 2:

```bash
python 4A-train-dgae-corel.py \
    --data-dir training_data/corel/corel_all \
    --lora-dir corel_models \
    --output-dir dgae_models
```

#### Training Parameters:

- `--data-dir`: Dataset directory **[REQUIRED]**
- `--lora-dir`: Directory with LoRA weights from Task 2 **[REQUIRED]**
- `--lora-name`: Specific LoRA file (default: use most recent)
- `--output-dir`: Output directory for DGAE model (default: `dgae_models`)
- `--base-dir`: Base directory for paths (default: `.`)
- `--pretrained-model`: Base Stable Diffusion model (default: `runwayml/stable-diffusion-v1-5`)
- `--epochs`: Number of training epochs (default: `300`)
- `--batch-size`: Training batch size (default: `8`, smaller due to diffusion model memory)
- `--latent-dim`: Latent dimension (default: `128`, compact for DGAE)
- `--learning-rate`: Learning rate (default: `1e-4`)
- `--diffusion-guidance-weight`: Weight for diffusion guidance loss (default: `0.1`)
- `--recon-weight`: Weight for reconstruction loss (default: `1.0`)
- `--no-perceptual`: Disable perceptual loss
- `--perceptual-weight`: Perceptual loss weight (default: `0.03`)
- `--no-diffusion-guidance`: Disable diffusion guidance (train regular autoencoder)
- `--num-workers`: DataLoader workers (default: `4`)
- `--resume`: Resume training from checkpoint
- `--seed`: Random seed (default: `42`)

#### Examples:

```bash
# Train with all classes
python 4A-train-dgae-corel.py \
    --data-dir training_data/corel/corel_all \
    --lora-dir corel_models \
    --output-dir dgae_models

# Train for specific class
python 4A-train-dgae-corel.py \
    --data-dir training_data/corel/class_0001 \
    --lora-dir corel_models \
    --lora-name lora_...class_0001...safetensors \
    --output-dir dgae_models/class_0001

# Custom parameters (more epochs, smaller batch)
python 4A-train-dgae-corel.py \
    --data-dir training_data/corel/corel_all \
    --lora-dir corel_models \
    --epochs 400 \
    --batch-size 4 \
    --latent-dim 256

# Train without diffusion guidance (regular autoencoder)
python 4A-train-dgae-corel.py \
    --data-dir training_data/corel/corel_all \
    --lora-dir corel_models \
    --no-diffusion-guidance
```

**Output:**
- `dgae_models/best_model.pt` - Best model checkpoint
- `dgae_models/checkpoint_*.pt` - Periodic checkpoints
- `dgae_models/samples/` - Generated samples
- `dgae_models/reconstructions/` - Reconstruction visualizations
- `dgae_models/training_losses.png` - Loss curves

### Step 2: Extract Latent Features for Clustering

Extract latent features from the trained DGAE model:

```bash
python 4B-extract-features-corel.py \
    --model-checkpoint dgae_models/best_model.pt \
    --data-dir training_data/corel/corel_all \
    --output features/dgae_features.npy
```

#### Extraction Parameters:

- `--model-checkpoint`: Path to DGAE model checkpoint **[REQUIRED]**
- `--data-dir`: Dataset directory **[REQUIRED]**
- `--output`: Output path for features (.npy file) **[REQUIRED]**
- `--base-dir`: Base directory for paths (default: `.`)
- `--batch-size`: Batch size for extraction (default: `32`)
- `--num-workers`: DataLoader workers (default: `4`)
- `--extract-labels`: Extract class labels from filenames (format: XXXX_YYYY.png)

#### Examples:

```bash
# Extract features with labels
python 4B-extract-features-corel.py \
    --model-checkpoint dgae_models/best_model.pt \
    --data-dir training_data/corel/corel_all \
    --output features/dgae_features.npy \
    --extract-labels

# Extract with custom batch size
python 4B-extract-features-corel.py \
    --model-checkpoint dgae_models/best_model.pt \
    --data-dir training_data/corel/corel_all \
    --output features/dgae_features.npy \
    --batch-size 64
```

**Output:**
- `features/dgae_features.npy` - Feature matrix (N x latent_dim)
- `features/dgae_features.json` - Metadata (num_samples, feature_dim, etc.)
- `features/dgae_features.labels.npy` - Class labels (if `--extract-labels`)
- `features/dgae_features.class_mapping.json` - Class number mapping
- `features/dgae_features.paths.txt` - Image file paths

## Using Features for Clustering

After extracting features, you can use them with standard clustering algorithms:

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Load features
features = np.load('features/dgae_features.npy')
labels = np.load('features/dgae_features.labels.npy')

# Perform clustering
kmeans = KMeans(n_clusters=len(set(labels)), random_state=42)
cluster_labels = kmeans.fit_predict(features)

# Evaluate clustering
ari = adjusted_rand_score(labels, cluster_labels)
nmi = normalized_mutual_info_score(labels, cluster_labels)

print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")
```

## Cloud Execution

All scripts support cloud execution via `--base-dir`:

```bash
# Example for cloud GPU rental
python 4A-train-dgae-corel.py \
    --data-dir /workspace/training_data/corel/corel_all \
    --lora-dir /workspace/corel_models \
    --output-dir /workspace/dgae_models \
    --base-dir /workspace

python 4B-extract-features-corel.py \
    --model-checkpoint /workspace/dgae_models/best_model.pt \
    --data-dir /workspace/training_data/corel/corel_all \
    --output /workspace/features/dgae_features.npy \
    --base-dir /workspace
```

## GPU Optimization

All scripts are optimized for GPU execution:

- **CUDA verification** - Checks GPU availability and displays info
- **CUDA optimizations** - `cudnn.benchmark=True` for faster convolutions
- **Memory optimizations** - Attention slicing, VAE tiling for diffusion model
- **Efficient feature extraction** - Batch processing with `pin_memory`
- **Periodic cache clearing** - Prevents memory accumulation

## DGAE vs Regular Autoencoder

**DGAE advantages:**
- Better reconstruction quality under high compression
- More compact latent space (2x smaller)
- Diffusion model guidance helps recover informative signals
- Better features for downstream tasks (clustering)

**When to use regular autoencoder:**
- If you don't have a pre-trained diffusion model
- If memory is extremely limited
- For baseline comparisons

## Complete Workflow Example

```bash
# 1. Ensure LoRA is trained (Task 2)
python 2B-train-lora-corel.py --data-dir training_data/corel/corel_all --output-dir corel_models

# 2. Train DGAE
python 4A-train-dgae-corel.py \
    --data-dir training_data/corel/corel_all \
    --lora-dir corel_models \
    --output-dir dgae_models \
    --epochs 300

# 3. Extract features for clustering
python 4B-extract-features-corel.py \
    --model-checkpoint dgae_models/best_model.pt \
    --data-dir training_data/corel/corel_all \
    --output features/dgae_features.npy \
    --extract-labels

# 4. Evaluate clustering (in Python)
# See "Using Features for Clustering" section above
```

## Troubleshooting

### Error: "No LoRA files found"
- Verify that Task 2 was completed successfully
- Check that `--lora-dir` points to the correct directory
- Ensure LoRA files have `.safetensors` extension

### Error: "CUDA out of memory"
- Reduce `--batch-size` (try 4 or 2)
- Use `--no-diffusion-guidance` to disable diffusion model (regular autoencoder)
- Reduce `--latent-dim`
- Enable gradient checkpointing (if implemented)

### Error: "Diffusion guidance loss computation failed"
- This is a warning, training continues without diffusion guidance
- Check that LoRA weights are compatible with the base model
- Try reducing `--diffusion-guidance-weight` or disable with `--no-diffusion-guidance`

### Error: "Model checkpoint not found"
- Verify the checkpoint path is correct
- Check that training completed successfully
- Ensure you're using `best_model.pt` or a valid checkpoint

### Poor reconstruction quality
- Increase `--epochs` (try 400-500)
- Increase `--diffusion-guidance-weight` (try 0.2-0.5)
- Ensure LoRA was trained well in Task 2
- Check that dataset has sufficient samples

## Output Structure

After training:
```
dgae_models/
├── best_model.pt              # Best model checkpoint
├── checkpoint_0020.pt         # Periodic checkpoints
├── checkpoint_0040.pt
├── ...
├── training_losses.png        # Loss curves
├── samples/                   # Generated samples
│   └── samples_epoch_*.png
└── reconstructions/           # Reconstruction visualizations
    └── reconstruction_epoch_*.png
```

After feature extraction:
```
features/
├── dgae_features.npy          # Feature matrix (N x latent_dim)
├── dgae_features.json         # Metadata
├── dgae_features.labels.npy   # Class labels (if extracted)
├── dgae_features.class_mapping.json  # Class mapping
└── dgae_features.paths.txt    # Image paths
```

## References

- **DGAE Paper**: [DGAE: Diffusion-Guided Autoencoder for Efficient Latent Representation Learning](https://arxiv.org/abs/2506.09644)
- **Task 2**: LoRA training (prerequisite)
- **Project description**: `description.pdf` (Task 4)

## Notes

- **Memory usage**: DGAE training requires significant GPU memory due to the diffusion model. Use smaller batch sizes if needed.
- **Training time**: DGAE training is slower than regular autoencoder due to diffusion guidance computation.
- **Clustering focus**: Remember that the goal is learning features for clustering, NOT data augmentation.
- **Per-class vs unified**: You can train per-class DGAE models similar to Task 2, but unified models are usually sufficient for clustering.

