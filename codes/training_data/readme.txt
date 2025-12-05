# Training Data Preparation Scripts

These scripts are OPTIONAL if you use the main preparation script (2A-prepare_corel_dataset.py).
The main script already generates captions.json automatically.

## Scripts

### generate_captions.py
Generates captions.json from Corel classification images.
**Note:** Already covered by `2A-prepare_corel_dataset.py`

### generate_balanced_augmentations.py
Generates dataset augmentations (useful for training VAE or models from scratch - codes 4 and 5).

## Usage

### Generate Captions (if not using 2A script)

```bash
# For a specific class
python generate_captions.py training_data/corel/class_0001

# For all classes
for class_dir in training_data/corel/class_*; do
    python generate_captions.py "$class_dir"
done

# Cloud execution
python generate_captions.py training_data/corel/class_0001 --base-dir /workspace
```

### Generate Augmentations (for VAE training)

```bash
# Local execution
python generate_balanced_augmentations.py \
    training_data/corel/class_0001 \
    augmented/class_0001 \
    --augmentations-per-image 25 \
    --mode balanced \
    --workers auto

# Cloud execution
python generate_balanced_augmentations.py \
    /workspace/data/corel \
    /workspace/augmented_corel \
    --base-dir /workspace \
    --augmentations-per-image 25 \
    --workers auto

# For all classes
for class_dir in training_data/corel/class_*; do
    python generate_balanced_augmentations.py \
        "$class_dir" \
        "augmented/$(basename $class_dir)" \
        --augmentations-per-image 25 \
        --mode balanced \
        --workers auto
done
```

## Dependencies

### generate_captions.py
No additional dependencies (uses standard library)

### generate_balanced_augmentations.py
```bash
pip install albumentations opencv-python tqdm
```

## Notes

- **generate_captions.py**: Redundant if using `2A-prepare_corel_dataset.py`
- **generate_balanced_augmentations.py**: Useful for augmenting data before training VAE or models from scratch
- Both scripts support local and cloud execution
- Both scripts support relative and absolute paths
