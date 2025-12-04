#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Balanced Dataset Augmentation for VAE Training
===============================================

Conservative augmentation that preserves color distribution and image quality.
Focuses on geometric transformations while keeping colors intact.

Ideal for datasets where color accuracy matters (e.g., uniforms, brands, products).

Usage:
    python generate_balanced_augmentations.py input_dir output_dir --augmentations-per-image 20
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')


def get_balanced_augmentation_pipeline(image_size=128):
    """
    Balanced augmentation pipeline:
    - STRONG geometric transformations (rotation, flip, shift, crop)
    - MINIMAL color changes (preserve color distribution)
    - NO noise, blur, or distortions that harm quality
    
    Perfect for datasets where color accuracy is important.
    """
    return A.Compose([
        # === GEOMETRIC TRANSFORMATIONS (AGGRESSIVE) ===
        # These create valid new perspectives without changing colors
        
        # Flips (free variations)
        A.HorizontalFlip(p=0.5),
        
        # Rotation (mild, natural angles)
        A.Rotate(limit=15, p=0.6, border_mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_LANCZOS4),
        
        # Small shifts and scaling
        A.ShiftScaleRotate(
            shift_limit=0.08,      # Small shifts
            scale_limit=0.15,      # Mild zoom in/out
            rotate_limit=10,       # Additional rotation
            p=0.6,
            border_mode=cv2.BORDER_REFLECT,
            interpolation=cv2.INTER_LANCZOS4
        ),
        
        # Perspective changes (subtle)
        A.Perspective(scale=(0.02, 0.05), p=0.3, interpolation=cv2.INTER_LANCZOS4),
        
        # Random cropping (creates new compositions)
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),     # Keep most of the image
            ratio=(0.95, 1.05),    # Almost square
            p=0.4,
            interpolation=cv2.INTER_LANCZOS4
        ),
        
        # === COLOR TRANSFORMATIONS (MINIMAL) ===
        # Only slight adjustments to mimic lighting changes
        
        # Slight brightness/contrast (like different lighting)
        A.RandomBrightnessContrast(
            brightness_limit=0.15,  # Very conservative
            contrast_limit=0.15,
            p=0.4
        ),
        
        # Very subtle color temperature shifts
        A.OneOf([
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02,  # Barely noticeable
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,   # Very small hue shifts
                sat_shift_limit=10,
                val_shift_limit=10,
                p=1.0
            ),
        ], p=0.3),
        
        # === QUALITY PRESERVATION ===
        # Slight sharpening to compensate for any softness
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.2),
        
        # Ensure final size (high-quality interpolation)
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
    ])


def get_conservative_augmentation_pipeline(image_size=128):
    """
    Even more conservative: only geometric transformations, ZERO color changes.
    Use this if color preservation is absolutely critical.
    """
    return A.Compose([
        # Only geometric transformations
        A.HorizontalFlip(p=0.5),
        
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_LANCZOS4),
        
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=5,
            p=0.5,
            border_mode=cv2.BORDER_REFLECT,
            interpolation=cv2.INTER_LANCZOS4
        ),
        
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.9, 1.0),
            ratio=(0.98, 1.02),
            p=0.3,
            interpolation=cv2.INTER_LANCZOS4
        ),
        
        # Final resize
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
    ])


def process_single_image(args):
    """Process a single image with augmentation (for parallel processing)"""
    img_path, output_dir, augmentations_per_image, image_size, save_quality, mode = args
    
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            return f"Failed to load: {img_path}"
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get augmentation pipeline based on mode
        if mode == 'conservative':
            transform = get_conservative_augmentation_pipeline(image_size)
        else:  # balanced
            transform = get_balanced_augmentation_pipeline(image_size)
        
        # Generate augmented versions
        img_name = img_path.stem
        results = []
        
        # Save original (resized)
        original_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
        original_path = output_dir / f"{img_name}_original.png"
        cv2.imwrite(
            str(original_path), 
            cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 9 - save_quality]
        )
        results.append(original_path)
        
        # Generate augmented versions
        for aug_idx in range(augmentations_per_image):
            augmented = transform(image=img)['image']
            
            aug_path = output_dir / f"{img_name}_aug{aug_idx:04d}.png"
            cv2.imwrite(
                str(aug_path), 
                cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 9 - save_quality]
            )
            results.append(aug_path)
        
        return f"✓ {img_name}: {len(results)} images"
    
    except Exception as e:
        return f"✗ Error processing {img_path}: {str(e)}"


def augment_dataset(input_dir, output_dir, augmentations_per_image=20, 
                   image_size=128, save_quality=9, num_workers=4, mode='balanced'):
    """
    Generate augmented dataset from input images.
    
    Args:
        input_dir: Path to original images
        output_dir: Path to save augmented images
        augmentations_per_image: Number of augmented versions per original image
        image_size: Target image size
        save_quality: PNG compression (0-9, where 9 is best quality)
        num_workers: Number of parallel workers
        mode: 'balanced' or 'conservative' augmentation strategy
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
        image_paths.extend(list(input_dir.rglob(ext)))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {input_dir}")
    
    print("="*80)
    print("BALANCED DATASET AUGMENTATION")
    print("="*80)
    print(f"Input directory:              {input_dir}")
    print(f"Output directory:             {output_dir}")
    print(f"Original images found:        {len(image_paths)}")
    print(f"Augmentations per image:      {augmentations_per_image}")
    print(f"Expected total images:        {len(image_paths) * (augmentations_per_image + 1)}")
    print(f"Target image size:            {image_size}x{image_size}")
    print(f"Augmentation mode:            {mode.upper()}")
    print(f"Parallel workers:             {num_workers}")
    print("="*80)
    
    if mode == 'balanced':
        print("\n📋 BALANCED MODE:")
        print("   ✓ Strong geometric transformations (rotation, flip, shift, crop)")
        print("   ✓ Minimal color changes (preserve color distribution)")
        print("   ✓ No noise, blur, or quality-degrading effects")
    else:
        print("\n📋 CONSERVATIVE MODE:")
        print("   ✓ Only geometric transformations")
        print("   ✓ ZERO color changes - perfect color preservation")
        print("   ✓ Ideal for color-critical applications")
    print()
    
    # Prepare arguments for parallel processing
    process_args = [
        (img_path, output_dir, augmentations_per_image, image_size, save_quality, mode)
        for img_path in image_paths
    ]
    
    # Process images in parallel
    print("Generating augmented images...\n")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_image, process_args),
            total=len(image_paths),
            desc="Processing images"
        ))
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    for result in results:
        print(result)
    
    # Count generated images
    generated_images = list(output_dir.glob("*.png"))
    
    print("\n" + "="*80)
    print(f"✅ Augmentation complete!")
    print(f"✅ Total images generated: {len(generated_images)}")
    print(f"✅ Expansion factor: {len(generated_images) / len(image_paths):.1f}x")
    print(f"✅ Output directory: {output_dir}")
    print("="*80)
    
def main():
    parser = argparse.ArgumentParser(
        description='Balanced augmentation that preserves color distribution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Balanced mode (recommended) - geometric + minimal color changes
  python generate_balanced_augmentations.py training_data/royalguard augmented_royalguard
  
  # Conservative mode - ONLY geometric, perfect color preservation
  python generate_balanced_augmentations.py images/ augmented/ --mode conservative
  
  # More aggressive augmentation count
  python generate_balanced_augmentations.py images/ augmented/ --augmentations-per-image 30
  
  # Fast processing with more workers
  python generate_balanced_augmentations.py images/ augmented/ --workers 8

Modes:
  balanced     - Strong geometric + minimal color changes (recommended)
  conservative - Only geometric transformations, zero color changes
        """
    )
    
    parser.add_argument('input_dir', type=str, 
                       help='Directory containing original images')
    parser.add_argument('output_dir', type=str, 
                       help='Directory to save augmented images')
    parser.add_argument('--augmentations-per-image', type=int, default=20,
                       help='Number of augmented versions per original image (default: 20)')
    parser.add_argument('--image-size', type=int, default=128,
                       help='Target image size (default: 128)')
    parser.add_argument('--quality', type=int, default=9, choices=range(0, 10),
                       help='PNG compression quality 0-9, where 9 is best (default: 9)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--mode', type=str, default='balanced',
                       choices=['balanced', 'conservative'],
                       help='Augmentation mode: balanced (geo+minimal color) or conservative (geo only)')
    
    args = parser.parse_args()
    
    # Run augmentation
    augment_dataset(
        args.input_dir,
        args.output_dir,
        augmentations_per_image=args.augmentations_per_image,
        image_size=args.image_size,
        save_quality=args.quality,
        num_workers=args.workers,
        mode=args.mode
    )


if __name__ == "__main__":
    main()
