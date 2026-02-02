#!/usr/bin/env python3
"""
Visualize COCOA Ground Truth Annotations
Check that dataset loading is working correctly
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'amodal_detection_head'))

from cocoa_dataset import COCOAAmodalDataset


def denormalize_image(image_tensor):
    """Denormalize image from ImageNet stats"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    return image


def box_cxcywh_to_xyxy(boxes, img_size):
    """
    Convert boxes from [cx, cy, w, h] normalized to [x1, y1, x2, y2] pixel coords
    
    Args:
        boxes: [N, 4] tensor in [cx, cy, w, h] format, normalized [0, 1]
        img_size: int, image size (assumed square)
    
    Returns:
        [N, 4] tensor in [x1, y1, x2, y2] format, pixel coordinates
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    
    return torch.stack([x1, y1, x2, y2], dim=1)


def visualize_ground_truth(sample, idx, output_dir='data_check'):
    """
    Visualize ground truth annotations
    
    Args:
        sample: Sample from dataset
        idx: Sample index
        output_dir: Directory to save visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data
    image = sample['image']
    visible_boxes = sample['visible_boxes']
    amodal_boxes = sample['amodal_boxes']
    occlusion_scores = sample['occlusion_scores']
    valid_mask = sample['valid_mask']
    file_name = sample.get('file_name', f'image_{idx}.jpg')
    
    # Denormalize image
    img_np = denormalize_image(image)
    img_size = image.shape[-1]
    
    # Get valid boxes
    valid_indices = torch.where(valid_mask > 0)[0]
    num_people = len(valid_indices)
    
    if num_people == 0:
        print(f"⚠️  Sample {idx}: No people found!")
        return
    
    # Convert to pixel coordinates
    vis_boxes_px = box_cxcywh_to_xyxy(visible_boxes[valid_indices], img_size)
    amod_boxes_px = box_cxcywh_to_xyxy(amodal_boxes[valid_indices], img_size)
    occ_scores = occlusion_scores[valid_indices]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img_np)
    ax.set_title(f'Sample {idx}: {num_people} person(s) - Ground Truth\n{Path(file_name).name}', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Draw boxes
    for i in range(num_people):
        # Amodal box (RED, DASHED) - Full person extent
        x1, y1, x2, y2 = amod_boxes_px[i]
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=3, edgecolor='red', facecolor='none',
            linestyle='--', label='Amodal (full extent)' if i == 0 else ''
        )
        ax.add_patch(rect)
        
        # Visible box (GREEN, SOLID) - What you can see
        x1, y1, x2, y2 = vis_boxes_px[i]
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=3, edgecolor='lime', facecolor='none',
            label='Visible (what you see)' if i == 0 else ''
        )
        ax.add_patch(rect)
        
        # Show occlusion score
        occ = occ_scores[i].item()
        ax.text(x1, y1 - 10, f'Person {i+1}\nOcc: {occ:.2f}',
               color='white', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=12)
    
    # Add interpretation text
    info_text = (
        f"🟢 GREEN (solid) = Visible parts\n"
        f"🔴 RED (dashed) = Amodal (full person)\n"
        f"Occlusion score: 0.0 = not occluded, 1.0 = fully hidden"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'gt_sample_{idx:03d}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Sample {idx}: {num_people} person(s), Occ range: [{occ_scores.min():.2f}, {occ_scores.max():.2f}]")
    print(f"   Saved: {output_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize COCOA Ground Truth')
    parser.add_argument('--coco-root', default='coco/train2014',
                        help='Path to COCO images')
    parser.add_argument('--cocoa-ann', default='coco/COCO_amodal_train2014_detectron.json',
                        help='Path to COCOA annotation JSON')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to visualize')
    parser.add_argument('--output-dir', default='data_check',
                        help='Output directory for visualizations')
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 VISUALIZING COCOA GROUND TRUTH ANNOTATIONS")
    print("="*80)
    print(f"📂 COCO images: {args.coco_root}")
    print(f"📂 COCOA annotations: {args.cocoa_ann}")
    print(f"📊 Samples to visualize: {args.num_samples}")
    print("="*80)
    
    # Load dataset
    print("\n📦 Loading COCOA dataset...")
    dataset = COCOAAmodalDataset(
        coco_root=args.coco_root,
        cocoa_annotation_file=args.cocoa_ann,
        split='train',
        image_size=640,
        max_objects=50,
        augment=False,
        person_only=True
    )
    
    if len(dataset) == 0:
        print("❌ No samples found in dataset!")
        return
    
    print(f"\n✅ Dataset loaded: {len(dataset)} images")
    print(f"\n🎨 Creating visualizations...")
    print("-"*80)
    
    # Visualize samples
    num_samples = min(args.num_samples, len(dataset))
    
    # Try to find samples with varying occlusion levels
    samples_to_viz = []
    
    # Get some random samples
    import random
    indices = random.sample(range(len(dataset)), min(num_samples * 3, len(dataset)))
    
    for idx in indices:
        sample = dataset[idx]
        num_valid = sample['valid_mask'].sum().item()
        
        if num_valid > 0:
            samples_to_viz.append((idx, sample))
            if len(samples_to_viz) >= num_samples:
                break
    
    # Visualize
    for i, (idx, sample) in enumerate(samples_to_viz):
        print(f"\n📸 Visualizing sample {i+1}/{len(samples_to_viz)} (dataset index {idx})...")
        visualize_ground_truth(sample, idx, args.output_dir)
    
    print("\n" + "="*80)
    print(f"✅ Visualization complete!")
    print(f"📁 Check images in: {args.output_dir}/")
    print("="*80)
    
    print("\n💡 What to look for:")
    print("   ✅ Green boxes should tightly fit visible person parts")
    print("   ✅ Red boxes should be larger, showing full person extent")
    print("   ✅ Occlusion score should be higher when more person is hidden")
    print("   ✅ Both boxes should align reasonably well")
    print("\n   ❌ If boxes are wildly off or both identical → data loading issue")
    print("   ❌ If images don't load → check file paths")


if __name__ == '__main__':
    main()