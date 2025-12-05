#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate captions.json for Corel Dataset
========================================

Script to generate captions.json from corel classification images.
Images are named as x_y.png where x is class number and y is example
number. Classes are defined in classes.txt with format: class_number class_name

This script is compatible with local and cloud execution (Colab, etc.)

Usage:
    python generate_captions.py folder_path
    python generate_captions.py training_data/corel/class_0001
"""

import os
import json
import glob
import re
from pathlib import Path
import sys


def read_classes(classes_file):
    """
    Read the classes.txt file and create a mapping from class number to class name.
    
    Args:
        classes_file: Path to classes.txt file
        
    Returns:
        Dictionary mapping class number (int) to class name (str)
    """
    class_mapping = {}
    
    with open(classes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Split by whitespace and take first column as class number
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                class_num = int(parts[0])
                class_name = parts[1].strip()
                class_mapping[class_num] = class_name
    
    return class_mapping


def generate_captions(folder_path):
    """
    Generate captions.json file from images in the given folder.
    
    Args:
        folder_path: Path to folder containing images and classes.txt
    """
    folder_path = Path(folder_path)
    
    # Read class mappings
    classes_file = folder_path / 'classes.txt'
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found in {folder_path}")
    
    class_mapping = read_classes(classes_file)
    print(f"Loaded {len(class_mapping)} classes from classes.txt")
    
    # Find all PNG images matching pattern x_y.png
    captions = {}
    pattern = re.compile(r'^(\d+)_(\d+)\.png$')
    
    png_files = sorted(folder_path.glob('*.png'))
    matched_count = 0
    
    for png_file in png_files:
        filename = png_file.name
        match = pattern.match(filename)
        
        if match:
            class_num = int(match.group(1))
            example_num = match.group(2)
            
            if class_num in class_mapping:
                captions[filename] = class_mapping[class_num]
                matched_count += 1
            else:
                print(f"Warning: Class {class_num} not found in classes.txt for {filename}")
    
    print(f"Processed {matched_count} images matching pattern x_y.png")
    
    # Save to captions.json
    output_file = folder_path / 'captions.json'
    with open(output_file, 'w') as f:
        json.dump(captions, f, indent=2)
    
    print(f"Captions saved to {output_file}")
    print(f"Total captions generated: {len(captions)}")


def main():
    """Main function to run the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate captions.json from corel classification images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local execution
  python generate_captions.py training_data/corel/class_0001
  
  # Cloud execution (Colab, etc.)
  python generate_captions.py /content/drive/MyDrive/training_data/corel/class_0001
  
  # Generate for all classes
  for class_dir in training_data/corel/class_*; do
    python generate_captions.py "$class_dir"
  done
        """
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Path to folder containing images and classes.txt'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Base directory (for cloud compatibility). If not set, uses folder as-is.'
    )
    
    args = parser.parse_args()
    
    # Handle base directory for cloud compatibility
    if args.base_dir:
        folder_path = Path(args.base_dir) / args.folder
    else:
        folder_path = Path(args.folder)
    
    # Resolve relative paths
    if not folder_path.is_absolute():
        folder_path = Path.cwd() / folder_path
    
    folder_path = folder_path.resolve()
    
    print("="*60)
    print("GENERATE CAPTIONS FOR COREL DATASET")
    print("="*60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Target folder: {folder_path}")
    print("="*60)
    
    if not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory")
        return 1
    
    try:
        generate_captions(folder_path)
        print("\n" + "="*60)
        print("✅ SUCCESS!")
        print("="*60)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
