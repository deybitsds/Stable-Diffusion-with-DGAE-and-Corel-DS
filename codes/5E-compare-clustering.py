#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Clustering Results for Task 5

This script compares clustering results from different feature extraction techniques:
- SimCLR (5A-simclr_corel.py)
- CNN-JEPA (5D-cnn-jepa_corel.py)
- DGAE (4B-extract-features-corel.py)

It evaluates clustering quality with and without diffusion augmentation.

Usage:
    python 5E-compare-clustering.py --features-dir features --output-dir clustering_results
"""

import numpy as np
from pathlib import Path
import json
import argparse
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import pandas as pd
except ImportError:
    print("⚠ Warning: pandas not installed. Install with: pip install pandas")
    print("  CSV export will be disabled, but JSON export will still work.")
    pd = None


def load_features(features_path):
    """Load features and labels from numpy files"""
    features_path = Path(features_path)
    
    # Load features
    features = np.load(features_path)
    
    # Load labels
    labels_path = features_path.with_suffix('.labels.npy')
    if labels_path.exists():
        labels = np.load(labels_path)
    else:
        print(f"⚠ Warning: Labels not found for {features_path.name}")
        labels = None
    
    # Load metadata
    metadata_path = features_path.with_suffix('.json')
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return features, labels, metadata


def evaluate_clustering(features, labels, method_name, n_clusters=None):
    """Evaluate clustering quality for a given feature set"""
    if labels is None:
        print(f"⚠ Warning: No labels available for {method_name}, skipping evaluation")
        return None
    
    # Get valid labels
    valid_mask = labels >= 0
    if valid_mask.sum() == 0:
        print(f"⚠ Warning: No valid labels for {method_name}")
        return None
    
    valid_labels = labels[valid_mask]
    valid_features = features[valid_mask]
    
    # Determine number of clusters
    if n_clusters is None:
        unique_labels = np.unique(valid_labels)
        n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        print(f"⚠ Warning: Not enough clusters for {method_name}")
        return None
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(valid_features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(features_scaled)
    
    # Calculate metrics
    metrics = {
        'method': method_name,
        'n_samples': len(valid_features),
        'n_clusters': n_clusters,
        'feature_dim': features.shape[1],
        'ari': float(adjusted_rand_score(valid_labels, predicted_labels)),
        'nmi': float(normalized_mutual_info_score(valid_labels, predicted_labels)),
        'silhouette': float(silhouette_score(features_scaled, predicted_labels)),
        'homogeneity': float(homogeneity_score(valid_labels, predicted_labels)),
        'completeness': float(completeness_score(valid_labels, predicted_labels)),
        'v_measure': float(v_measure_score(valid_labels, predicted_labels)),
    }
    
    return metrics


def compare_all_techniques(features_dir, output_dir, with_augmentation=False):
    """Compare all feature extraction techniques"""
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CLUSTERING COMPARISON FOR TASK 5")
    print("="*60)
    print(f"Features directory: {features_dir}")
    print(f"Output directory: {output_dir}")
    print(f"With diffusion augmentation: {with_augmentation}")
    print("="*60 + "\n")
    
    # Define feature files to compare
    suffix = "_aug" if with_augmentation else ""
    
    feature_files = {
        'SimCLR': features_dir / f'simclr_features{suffix}.npy',
        'CNN-JEPA': features_dir / f'cnn_jepa_features{suffix}.npy',
        'DGAE': features_dir / f'dgae_features{suffix}.npy',
    }
    
    # Load and evaluate each technique
    all_metrics = []
    
    for method_name, features_path in feature_files.items():
        if not features_path.exists():
            print(f"⚠ Warning: Features not found for {method_name}: {features_path}")
            print(f"  Skipping {method_name}...\n")
            continue
        
        print(f"Evaluating {method_name}...")
        print(f"  Loading: {features_path}")
        
        try:
            features, labels, metadata = load_features(features_path)
            print(f"  Features shape: {features.shape}")
            print(f"  Labels shape: {labels.shape if labels is not None else 'None'}")
            
            metrics = evaluate_clustering(features, labels, method_name)
            
            if metrics:
                all_metrics.append(metrics)
                print(f"  ✓ ARI: {metrics['ari']:.4f}")
                print(f"  ✓ NMI: {metrics['nmi']:.4f}")
                print(f"  ✓ Silhouette: {metrics['silhouette']:.4f}")
            else:
                print(f"  ⚠ Could not evaluate {method_name}")
        
        except Exception as e:
            print(f"  ✗ Error evaluating {method_name}: {e}")
        
        print()
    
    if len(all_metrics) == 0:
        print("⚠ No metrics collected. Please check that feature files exist.")
        return None
    
    # Save results
    results_path = output_dir / f'clustering_comparison{suffix}.json'
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Saved results to: {results_path}")
    
    # Create comparison DataFrame if pandas is available
    if pd is not None:
        df = pd.DataFrame(all_metrics)
        
        # Save CSV
        csv_path = output_dir / f'clustering_comparison{suffix}.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV to: {csv_path}")
        
        # Print comparison table
        print("\n" + "="*60)
        print("CLUSTERING COMPARISON RESULTS")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)
        
        # Find best method for each metric
        print("\n" + "="*60)
        print("BEST METHODS BY METRIC")
        print("="*60)
        
        metric_names = ['ari', 'nmi', 'silhouette', 'homogeneity', 'completeness', 'v_measure']
        for metric in metric_names:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_method = df.loc[best_idx, 'method']
                best_value = df.loc[best_idx, metric]
                print(f"  {metric.upper():<15}: {best_method:<15} ({best_value:.4f})")
        
        # Create visualization
        create_comparison_plot(df, output_dir / f'clustering_comparison{suffix}.png', suffix)
    else:
        # Print results without pandas
        print("\n" + "="*60)
        print("CLUSTERING COMPARISON RESULTS")
        print("="*60)
        for metrics in all_metrics:
            print(f"\n{metrics['method']}:")
            print(f"  ARI: {metrics['ari']:.4f}")
            print(f"  NMI: {metrics['nmi']:.4f}")
            print(f"  Silhouette: {metrics['silhouette']:.4f}")
        print("="*60)
    
    return all_metrics


def create_comparison_plot(df, output_path, suffix=""):
    """Create visualization comparing all techniques"""
    if df is None or len(df) == 0:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    metrics_to_plot = ['ari', 'nmi', 'silhouette', 'homogeneity', 'completeness', 'v_measure']
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            ax = axes[i]
            bars = ax.bar(df['method'], df[metric], alpha=0.7, edgecolor='black')
            ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
            
            # Rotate x-axis labels if needed
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle(f'Clustering Comparison{" (with diffusion augmentation)" if suffix else " (without augmentation)"}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare Clustering Results from Different Techniques',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare without augmentation
  python 5E-compare-clustering.py --features-dir features --output-dir clustering_results
  
  # Compare with augmentation
  python 5E-compare-clustering.py --features-dir features --output-dir clustering_results --with-augmentation
  
  # Compare both
  python 5E-compare-clustering.py --features-dir features --output-dir clustering_results --compare-both
        """
    )
    
    parser.add_argument(
        '--features-dir',
        type=str,
        default='features',
        help='Directory containing feature files (default: features)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='clustering_results',
        help='Output directory for comparison results (default: clustering_results)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='.',
        help='Base directory for paths (default: current directory)'
    )
    parser.add_argument(
        '--with-augmentation',
        action='store_true',
        help='Compare features with diffusion augmentation (looks for *_aug.npy files)'
    )
    parser.add_argument(
        '--compare-both',
        action='store_true',
        help='Compare both with and without augmentation'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir).resolve()
    features_dir = base_dir / args.features_dir if not Path(args.features_dir).is_absolute() else Path(args.features_dir)
    features_dir = features_dir.resolve()
    output_dir = base_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir = output_dir.resolve()
    
    if not features_dir.exists():
        print(f"ERROR: Features directory not found: {features_dir}")
        return 1
    
    # Compare techniques
    if args.compare_both:
        print("\n" + "="*60)
        print("COMPARING WITHOUT AUGMENTATION")
        print("="*60)
        compare_all_techniques(features_dir, output_dir, with_augmentation=False)
        
        print("\n" + "="*60)
        print("COMPARING WITH AUGMENTATION")
        print("="*60)
        compare_all_techniques(features_dir, output_dir, with_augmentation=True)
    else:
        compare_all_techniques(features_dir, output_dir, with_augmentation=args.with_augmentation)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

