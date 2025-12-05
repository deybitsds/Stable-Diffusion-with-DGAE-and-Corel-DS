# Task 5: Clustering Evaluation with Self-Supervised Learning Techniques

This guide covers evaluating clustering of Corel dataset images using different self-supervised learning techniques for feature extraction.

Based on Task 5 from `description.pdf`: Evaluate clustering with and without diffusion augmentation using:
- **Self-supervised technique**: SimCLR (chosen for simplicity and speed)
- **CNN-JEPA** [2]
- **DGAE** [3]

**Note**: The task allows choosing one technique from SimCLR, BYOL, DINO, or DINO-MultiCrop. We chose SimCLR as it is the simplest and fastest to run.

## Overview

Task 5 involves:
1. Training self-supervised models to learn visual representations
2. Extracting features from trained models
3. Evaluating clustering quality using metrics (ARI, NMI, Silhouette Score)
4. Comparing different techniques with and without diffusion augmentation

## Project Structure

```
codes/
├── 5A-simclr_corel.py          # SimCLR implementation (✓ Ready)
├── 5D-cnn-jepa_corel.py        # CNN-JEPA implementation (✓ Ready)
├── 5E-compare-clustering.py    # Compare all techniques (✓ Ready)
├── 4A-train-dgae-corel.py      # DGAE training (Task 4)
├── 4B-extract-features-corel.py # DGAE feature extraction (Task 4)
├── training_data/
│   └── corel/
│       ├── corel_all/          # All classes
│       └── class_XXXX/         # Per-class datasets
├── simclr_models/              # SimCLR trained models
│   └── best_model.pt
├── cnn_jepa_models/            # CNN-JEPA trained models
│   └── best_model.pt
├── features/                   # Extracted features for clustering
│   ├── simclr_features.npy
│   ├── cnn_jepa_features.npy
│   ├── dgae_features.npy
│   └── ...
└── clustering_results/         # Clustering evaluation results
    ├── clustering_comparison.json
    ├── clustering_comparison.csv
    └── clustering_comparison.png
```

## Installation

### Install Dependencies

```bash
# Core dependencies
pip install torch torchvision

# For self-supervised learning
pip install numpy scikit-learn matplotlib tqdm pandas

# For visualization and clustering
pip install umap-learn scipy seaborn

# Note: The package is 'umap-learn', not 'umap'
# Install with: pip install umap-learn

# Optional: for advanced clustering
pip install hdbscan
```

## Execution Order

### Step 1: Train Self-Supervised Models

#### Option A: SimCLR (Currently Available)

```bash
python 5A-simclr_corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir simclr_models \
    --epochs 100 \
    --batch-size 32
```

**Training Parameters:**
- `--data-dir`: Dataset directory **[REQUIRED]**
- `--output-dir`: Output directory for model (default: `simclr_models`)
- `--base-dir`: Base directory for paths (default: `.`)
- `--epochs`: Number of training epochs (default: `100`)
- `--batch-size`: Training batch size (default: `32`)
- `--image-size`: Image size for training (default: `224`)
- `--latent-dim`: Latent dimension for projection head (default: `128`)
- `--learning-rate`: Learning rate (default: `3e-3`)
- `--temperature`: Temperature parameter for SimCLR loss (default: `0.5`)
- `--weight-decay`: Weight decay (default: `1e-4`)
- `--num-workers`: DataLoader workers (default: `4`)
- `--resume`: Resume training from checkpoint
- `--seed`: Random seed (default: `42`)

**Examples:**

```bash
# Basic training
python 5A-simclr_corel.py --data-dir training_data/corel/corel_all --output-dir simclr_models

# Training with custom parameters
python 5A-simclr_corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir simclr_models \
    --epochs 150 \
    --batch-size 64 \
    --latent-dim 256 \
    --learning-rate 1e-3

# Train for specific class
python 5A-simclr_corel.py \
    --data-dir training_data/corel/class_0001 \
    --output-dir simclr_models/class_0001 \
    --epochs 200
```

#### Option B: DGAE (From Task 4)

```bash
# Train DGAE (requires LoRA from Task 2)
python 4A-train-dgae-corel.py \
    --data-dir training_data/corel/corel_all \
    --lora-dir corel_models \
    --output-dir dgae_models

# Extract DGAE features
python 4B-extract-features-corel.py \
    --model-checkpoint dgae_models/best_model.pt \
    --data-dir training_data/corel/corel_all \
    --output features/dgae_features.npy \
    --extract-labels
```

#### Option C: CNN-JEPA

```bash
python 5D-cnn-jepa_corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir cnn_jepa_models \
    --epochs 100
```

**Training Parameters:**
- `--data-dir`: Dataset directory **[REQUIRED]**
- `--output-dir`: Output directory for model (default: `cnn_jepa_models`)
- `--epochs`: Number of training epochs (default: `100`)
- `--batch-size`: Training batch size (default: `32`)
- `--image-size`: Image size for training (default: `224`)
- `--feature-dim`: Feature dimension (default: `512`)
- `--hidden-dim`: Hidden dimension for predictor (default: `512`)
- `--learning-rate`: Learning rate (default: `1e-3`)
- `--num-workers`: DataLoader workers (default: `4`)

**Examples:**

```bash
# Basic training
python 5D-cnn-jepa_corel.py --data-dir training_data/corel/corel_all --output-dir cnn_jepa_models

# Train and extract features
python 5D-cnn-jepa_corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir cnn_jepa_models \
    --extract-features features/cnn_jepa_features.npy \
    --evaluate-clustering
```

### Step 2: Extract Features for Clustering

#### SimCLR Feature Extraction

```bash
python 5A-simclr_corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir simclr_models \
    --extract-features features/simclr_features.npy \
    --evaluate-clustering
```

**Feature Extraction Parameters:**
- `--extract-features`: Output path for features (.npy file)
- `--use-projection`: Use projection head features instead of encoder features
- `--evaluate-clustering`: Evaluate clustering quality after extraction

**Output:**
- `features/simclr_features.npy` - Feature matrix (N x feature_dim)
- `features/simclr_features.json` - Metadata
- `features/simclr_features.labels.npy` - Class labels
- `features/simclr_features.paths.txt` - Image paths
- `simclr_models/clustering_metrics.json` - Clustering metrics (if `--evaluate-clustering`)

### Step 3: Compare All Techniques

Use the comparison script to evaluate all techniques at once:

```bash
# Compare without diffusion augmentation
python 5E-compare-clustering.py \
    --features-dir features \
    --output-dir clustering_results

# Compare with diffusion augmentation
python 5E-compare-clustering.py \
    --features-dir features \
    --output-dir clustering_results \
    --with-augmentation

# Compare both (with and without augmentation)
python 5E-compare-clustering.py \
    --features-dir features \
    --output-dir clustering_results \
    --compare-both
```

**Comparison Parameters:**
- `--features-dir`: Directory containing feature files (default: `features`)
- `--output-dir`: Output directory for results (default: `clustering_results`)
- `--with-augmentation`: Compare features with diffusion augmentation
- `--compare-both`: Compare both with and without augmentation

**Output:**
- `clustering_results/clustering_comparison.json` - Detailed metrics (JSON)
- `clustering_results/clustering_comparison.csv` - Metrics table (CSV)
- `clustering_results/clustering_comparison.png` - Visualization

#### Manual Evaluation (Alternative)

After extracting features, you can also evaluate clustering manually using Python:

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Load features
features = np.load('features/simclr_features.npy')
labels = np.load('features/simclr_features.labels.npy')

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Perform clustering
n_clusters = len(set(labels[labels >= 0]))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features_scaled)

# Evaluate clustering
valid_mask = labels >= 0
ari = adjusted_rand_score(labels[valid_mask], cluster_labels[valid_mask])
nmi = normalized_mutual_info_score(labels[valid_mask], cluster_labels[valid_mask])
silhouette = silhouette_score(features_scaled, cluster_labels)

print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")
```

#### Manual Comparison (Alternative)

You can also manually compare techniques using Python:

```python
import numpy as np
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

def evaluate_clustering(features_path, method_name):
    """Evaluate clustering for a given feature file"""
    features = np.load(features_path)
    labels_path = Path(features_path).with_suffix('.labels.npy')
    labels = np.load(labels_path)
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Cluster
    unique_labels = np.unique(labels[labels >= 0])
    n_clusters = len(unique_labels)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Metrics
    valid_mask = labels >= 0
    ari = adjusted_rand_score(labels[valid_mask], cluster_labels[valid_mask])
    nmi = normalized_mutual_info_score(labels[valid_mask], cluster_labels[valid_mask])
    silhouette = silhouette_score(features_scaled, cluster_labels)
    
    return {
        'method': method_name,
        'ari': float(ari),
        'nmi': float(nmi),
        'silhouette': float(silhouette),
        'n_clusters': int(n_clusters)
    }

# Compare all techniques
results = []

# SimCLR
results.append(evaluate_clustering('features/simclr_features.npy', 'SimCLR'))

# DGAE
results.append(evaluate_clustering('features/dgae_features.npy', 'DGAE'))

# Save comparison
with open('clustering_results/comparison_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "="*60)
print("CLUSTERING COMPARISON")
print("="*60)
print(f"{'Method':<15} {'ARI':<10} {'NMI':<10} {'Silhouette':<12}")
print("-" * 60)
for r in results:
    print(f"{r['method']:<15} {r['ari']:<10.4f} {r['nmi']:<10.4f} {r['silhouette']:<12.4f}")
```

## Clustering with and without Diffusion Augmentation

### Without Diffusion Augmentation

Train models directly on original Corel dataset:

```bash
# SimCLR on original data
python 5A-simclr_corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir simclr_models \
    --extract-features features/simclr_no_aug_features.npy
```

### With Diffusion Augmentation

Use images generated from Task 2 or Task 3 for training:

```bash
# Option 1: Use LoRA-generated images (Task 2)
# First generate augmented images using 2C-generate-lora-corel.py
# Then train on augmented dataset

# Option 2: Use diffusion-generated images (Task 3)
# First generate images using 3C-generate-samples-corel.py
# Then train on augmented dataset

# SimCLR on augmented data
python 5A-simclr_corel.py \
    --data-dir training_data/corel/corel_all_augmented \
    --output-dir simclr_models_aug \
    --extract-features features/simclr_aug_features.npy
```

## Cloud Execution

All scripts support cloud execution via `--base-dir`:

```bash
# Example for cloud GPU rental
python 5A-simclr_corel.py \
    --data-dir /workspace/training_data/corel/corel_all \
    --output-dir /workspace/simclr_models \
    --base-dir /workspace \
    --extract-features /workspace/features/simclr_features.npy
```

## GPU Optimization

All scripts are optimized for GPU execution:

- **CUDA verification** - Checks GPU availability and displays info
- **CUDA optimizations** - `cudnn.benchmark=True` for faster convolutions
- **Efficient data loading** - DataLoader with `pin_memory` and `num_workers`
- **Memory management** - Gradient clipping and periodic cache clearing

## Complete Workflow Example

```bash
# 1. Train SimCLR
python 5A-simclr_corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir simclr_models \
    --epochs 100 \
    --batch-size 32

# 2. Extract SimCLR features
python 5A-simclr_corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir simclr_models \
    --extract-features features/simclr_features.npy \
    --evaluate-clustering

# 3. Train CNN-JEPA
python 5D-cnn-jepa_corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir cnn_jepa_models \
    --epochs 100

# 4. Extract CNN-JEPA features
python 5D-cnn-jepa_corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir cnn_jepa_models \
    --extract-features features/cnn_jepa_features.npy \
    --evaluate-clustering

# 5. Extract DGAE features (if Task 4 completed)
python 4B-extract-features-corel.py \
    --model-checkpoint dgae_models/best_model.pt \
    --data-dir training_data/corel/corel_all \
    --output features/dgae_features.npy \
    --extract-labels

# 6. Compare all techniques
python 5E-compare-clustering.py \
    --features-dir features \
    --output-dir clustering_results \
    --compare-both
```

## Clustering Metrics Explained

### Adjusted Rand Index (ARI)
- **Range**: [-1, 1], higher is better
- **Meaning**: Measures agreement between true labels and cluster assignments
- **1.0**: Perfect clustering
- **0.0**: Random clustering
- **Negative**: Worse than random

### Normalized Mutual Information (NMI)
- **Range**: [0, 1], higher is better
- **Meaning**: Measures mutual information between true labels and clusters
- **1.0**: Perfect clustering
- **0.0**: No mutual information

### Silhouette Score
- **Range**: [-1, 1], higher is better
- **Meaning**: Measures how similar objects are to their own cluster vs other clusters
- **1.0**: Perfect separation
- **0.0**: Overlapping clusters
- **Negative**: Poor clustering

## Troubleshooting

### Error: "No images found"
- Verify that `--data-dir` points to the correct directory
- Check that images have valid extensions (.png, .jpg, .jpeg, .bmp)

### Error: "CUDA out of memory"
- Reduce `--batch-size` (try 16 or 8)
- Reduce `--image-size` (try 128 or 64)
- Reduce `--latent-dim`

### Error: "Model checkpoint not found"
- Verify that training completed successfully
- Check that you're using `best_model.pt` or a valid checkpoint path

### Poor clustering results
- Increase `--epochs` (try 150-200)
- Adjust `--temperature` for SimCLR (try 0.1-1.0)
- Ensure dataset has sufficient samples per class
- Try different `--latent-dim` values

### Feature extraction fails
- Ensure model was trained successfully
- Check that `--data-dir` matches training data directory
- Verify image paths are accessible

## Output Structure

After training SimCLR:
```
simclr_models/
├── best_model.pt              # Best model checkpoint
└── clustering_metrics.json    # Clustering metrics (if evaluated)
```

After feature extraction:
```
features/
├── simclr_features.npy        # Feature matrix
├── simclr_features.json       # Metadata
├── simclr_features.labels.npy # Class labels
└── simclr_features.paths.txt # Image paths
```

## Techniques Implemented

### Completed:
- **5A-simclr_corel.py**: SimCLR (Simple Contrastive Learning) ✓
- **5D-cnn-jepa_corel.py**: CNN-JEPA (Joint Embedding Predictive Architecture) ✓
- **5E-compare-clustering.py**: Compare all techniques and generate reports ✓
- **4B-extract-features-corel.py**: DGAE feature extraction (from Task 4) ✓

### Note on Technique Selection:
The task allows choosing one technique from SimCLR, BYOL, DINO, or DINO-MultiCrop. We chose **SimCLR** because:
- It is the simplest to implement and understand
- It is faster to train than other techniques
- It provides good baseline results for clustering evaluation

## References

- **SimCLR Paper**: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- **BYOL Paper**: [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733)
- **DINO Paper**: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- **CNN-JEPA Paper**: [CNN-JEPA: Self-Supervised Pretraining Convolutional Neural Networks Using Joint Embedding Predictive Architecture](https://arxiv.org/abs/2408.07514)
- **DGAE Paper**: [DGAE: Diffusion-Guided Autoencoder for Efficient Latent Representation Learning](https://arxiv.org/abs/2506.09644)
- **Project description**: `description.pdf` (Task 5)

## Notes

- **Training time**: SimCLR training can take several hours depending on dataset size and GPU
- **Feature extraction**: Fast once model is trained (minutes)
- **Clustering evaluation**: Very fast (seconds to minutes)
- **Comparison**: Essential for Task 5 to compare all techniques
- **Diffusion augmentation**: Can improve clustering by providing more diverse training data

