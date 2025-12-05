#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Corel Dataset for Stable Diffusion

This script prepares the Corel dataset for training Stable Diffusion models with LoRA.

Tasks:
1. Load and explore the Corel dataset
2. Generate metadata (captions.json) for each class
3. Organize data into appropriate folders
4. Prepare everything for training and generation

Usage:
    python 2A-prepare_corel_dataset.py --data-dir data/corel --base-dir .
"""

import os
import json
import glob
import re
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import shutil
import argparse
import sys


def read_classes(classes_file):
    """Read classes.txt file and return a dictionary class -> name"""
    class_mapping = {}
    
    with open(classes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                class_num = int(parts[0])
                class_name = parts[1].strip()
                class_mapping[class_num] = class_name
    
    return class_mapping


def generate_captions_for_class(class_num, class_name, image_files, output_dir):
    """Generate captions.json for a specific class"""
    class_dir = output_dir / f"class_{class_num:04d}"
    class_dir.mkdir(parents=True, exist_ok=True)
    
    captions = {}
    pattern = re.compile(r'^(\d+)_(\d+)\.png$')
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        match = pattern.match(filename)
        
        if match and int(match.group(1)) == class_num:
            # Copy image to class folder
            dest_path = class_dir / filename
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
            
            # Create descriptive caption
            caption = f"a photo of a {class_name}"
            captions[filename] = caption
    
    # Save captions.json
    captions_file = class_dir / 'captions.json'
    with open(captions_file, 'w') as f:
        json.dump(captions, f, indent=2)
    
    return len(captions), class_dir


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Corel Dataset for Stable Diffusion training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local execution
  python 2A-prepare_corel_dataset.py --data-dir data/corel --base-dir .
  
  # Cloud execution
  python 2A-prepare_corel_dataset.py --data-dir data/corel --base-dir /workspace
  
  # Custom output directories
  python 2A-prepare_corel_dataset.py --data-dir data/corel --base-dir . \\
      --output-dir custom_training_data --figs-dir custom_figs
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/corel',
        help='Directory containing Corel dataset images (default: data/corel)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='.',
        help='Base directory for paths (default: current directory)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for training data (default: base_dir/training_data)'
    )
    parser.add_argument(
        '--figs-dir',
        type=str,
        default=None,
        help='Directory for visualization figures (default: base_dir/figs)'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip dataset visualization'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir).resolve()
    corel_data_dir = base_dir / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    corel_data_dir = corel_data_dir.resolve()
    
    training_data_dir = base_dir / (args.output_dir or 'training_data')
    corel_training_dir = training_data_dir / 'corel'
    figs_dir = base_dir / (args.figs_dir or 'figs')
    
    # Create directories
    training_data_dir.mkdir(parents=True, exist_ok=True)
    corel_training_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("PREPARE COREL DATASET FOR STABLE DIFFUSION")
    print("="*60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Base directory: {base_dir}")
    print(f"Corel dataset: {corel_data_dir}")
    print(f"Training data: {corel_training_dir}")
    print(f"Figures: {figs_dir}")
    print("="*60)
    
    # Find all PNG images
    image_files = sorted(glob.glob(str(corel_data_dir / '*.png')))
    print(f"\nTotal images found: {len(image_files)}")
    
    if len(image_files) == 0:
        print(f"ERROR: No PNG files found in {corel_data_dir}")
        print("Please check the path configuration.")
        return 1
    
    # Analyze class structure
    class_distribution = defaultdict(list)
    pattern = re.compile(r'^(\d+)_(\d+)\.png$')
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        match = pattern.match(filename)
        if match:
            class_num = int(match.group(1))
            example_num = match.group(2)
            class_distribution[class_num].append(filename)
    
    # Show class distribution
    print(f"\nClasses found: {len(class_distribution)}")
    print("\nDistribution by class:")
    for class_num in sorted(class_distribution.keys()):
        count = len(class_distribution[class_num])
        print(f"  Class {class_num:04d}: {count} images")
    
    # Check if classes.txt exists, otherwise create a generic one
    classes_file = corel_data_dir / 'classes.txt'
    
    if classes_file.exists():
        print(f"\nFound classes.txt at {classes_file}")
        with open(classes_file, 'r') as f:
            print("\nContent:")
            print(f.read())
    else:
        print(f"\nWARNING: classes.txt not found. Creating a generic one...")
        
        # Create generic names based on class numbers
        class_names = {}
        for class_num in sorted(class_distribution.keys()):
            # Generic names - ADJUST according to your actual dataset
            generic_names = [
                "royalguard", "beach", "mountain", "flower",
                "building", "animal", "vehicle", "person"
            ]
            idx = (class_num - 1) % len(generic_names)
            class_names[class_num] = generic_names[idx]
        
        # Save classes.txt
        with open(classes_file, 'w') as f:
            for class_num in sorted(class_names.keys()):
                f.write(f"{class_num} {class_names[class_num]}\n")
        
        print(f"Created classes.txt at {classes_file}")
        print("⚠ IMPORTANT: Please edit classes.txt with actual class names before proceeding!")
    
    # Read classes
    class_mapping = read_classes(classes_file)
    print(f"\nLoaded {len(class_mapping)} classes from classes.txt")
    
    # Generate captions for each class
    print("\nGenerating captions.json for each class...")
    class_dirs = {}
    
    for class_num in sorted(class_distribution.keys()):
        if class_num in class_mapping:
            class_name = class_mapping[class_num]
            count, class_dir = generate_captions_for_class(
                class_num, class_name, image_files, corel_training_dir
            )
            class_dirs[class_num] = class_dir
            print(f"  Class {class_num:04d} ({class_name}): {count} images -> {class_dir.name}")
        else:
            print(f"  WARNING: Class {class_num:04d} not found in classes.txt")
    
    # Create unified dataset with all classes
    corel_all_dir = corel_training_dir / 'corel_all'
    corel_all_dir.mkdir(parents=True, exist_ok=True)
    
    all_captions = {}
    pattern = re.compile(r'^(\d+)_(\d+)\.png$')
    
    print("\nCreating unified dataset (all classes)...")
    for img_path in image_files:
        filename = os.path.basename(img_path)
        match = pattern.match(filename)
        
        if match:
            class_num = int(match.group(1))
            
            if class_num in class_mapping:
                class_name = class_mapping[class_num]
                
                # Copy image
                dest_path = corel_all_dir / filename
                if not dest_path.exists():
                    shutil.copy2(img_path, dest_path)
                
                # Create caption
                caption = f"a photo of a {class_name}"
                all_captions[filename] = caption
    
    # Save unified captions.json
    captions_file = corel_all_dir / 'captions.json'
    with open(captions_file, 'w') as f:
        json.dump(all_captions, f, indent=2)
    
    print(f"Unified dataset created: {corel_all_dir}")
    print(f"  Total images: {len(all_captions)}")
    print(f"  Captions saved to: {captions_file}")
    
    # Visualize dataset samples
    if not args.no_visualize:
        print("\nGenerating visualization...")
        n_samples_per_class = 3
        n_classes = len(class_distribution)
        
        fig, axes = plt.subplots(n_classes, n_samples_per_class, figsize=(15, 5*n_classes))
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        for class_idx, class_num in enumerate(sorted(class_distribution.keys())[:n_classes]):
            class_images = class_distribution[class_num][:n_samples_per_class]
            class_name = class_mapping.get(class_num, f"class_{class_num}")
            
            for img_idx, img_filename in enumerate(class_images):
                img_path = corel_data_dir / img_filename
                
                if img_path.exists():
                    img = Image.open(img_path).convert('RGB')
                    axes[class_idx, img_idx].imshow(img)
                    axes[class_idx, img_idx].set_title(f"{class_name}\n{img_filename}", fontsize=10)
                    axes[class_idx, img_idx].axis('off')
        
        plt.suptitle('Corel Dataset Samples by Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        vis_path = figs_dir / 'corel_dataset_samples.png'
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {vis_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PREPARATION SUMMARY")
    print("="*60)
    print(f"\nCorel dataset processed:")
    print(f"  - Total images: {len(image_files)}")
    print(f"  - Classes found: {len(class_distribution)}")
    print(f"\nStructure created:")
    print(f"  - Unified dataset: {corel_all_dir}")
    print(f"    -> {len(all_captions)} images with captions.json")
    print(f"\n  - Per-class datasets:")
    for class_num, class_dir in sorted(class_dirs.items()):
        class_name = class_mapping.get(class_num, 'unknown')
        img_count = len(list(class_dir.glob('*.png')))
        print(f"    -> class_{class_num:04d} ({class_name}): {img_count} images")
    
    print(f"\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. To train LoRA with ALL classes:")
    print(f"   python 2B-train-lora-corel.py --data-dir {corel_all_dir}")
    print("\n2. To train LoRA per class (recommended if few samples per class):")
    for class_num in sorted(class_dirs.keys())[:3]:  # Show only first 3
        class_dir = class_dirs[class_num]
        print(f"   python 2B-train-lora-corel.py --data-dir {class_dir}")
    if len(class_dirs) > 3:
        print(f"   ... and {len(class_dirs)-3} more classes")
    
    print("\n3. To generate images with trained LoRA:")
    print(f"   python 2C-generate-lora-corel.py --lora-dir {base_dir / 'corel_models'}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

