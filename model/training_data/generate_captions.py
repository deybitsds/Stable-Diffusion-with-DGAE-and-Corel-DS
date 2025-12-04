#!/usr/bin/env python3

"""Script to generate captions.json from corel classification images.
Images are named as x_y.png where x is class number and y is example
number.  Classes are defined in classes.txt with format: class_number
class_name

"""

import os
import json
import glob
import re
from pathlib import Path


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
        description='Generate captions.json from corel classification images'
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Path to folder containing images and classes.txt'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        return 1
    
    try:
        generate_captions(args.folder)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
