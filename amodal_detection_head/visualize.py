#!/usr/bin/env python3
"""
Visualize COCOA Dataset - Humans Only
Check data before training
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'amodal_detection_head'))

from cocoa_dataset import COCOAAmodalDataset


def denormalize_image(image_tensor):
    """Denormalize image tensor"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    return image


def xyxy_to_viz(boxes, img_size):
    """Convert normalized [x1, y1, x2, y2] to pixel coords for viz"""
    boxes_px = boxes.clone()
    boxes_px[:, [0, 2]] *= img_size
    boxes_px[:, [1, 3]] *= img_size
    return boxes_px


def visualize_sample(dataset, idx, output_dir='viz'):
    """Visualize one sample"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data
    sample = dataset[idx]
    image = sample['image']
    visible_boxes = sample['visible_boxes']
    amodal_boxes = sample['amodal_boxes']
    occlusion_scores = sample['occlusion_scores']
    valid_mask = sample['valid_mask']
    
    # Denormalize image
    img_np = denormalize_image(image)
    img_size = image.shape[-1]
    
    # Get valid objects
    valid_indices = torch.where(valid_mask > 0)[0]
    num_humans = len(valid_indices)
    
    if num_humans == 0:
        print(f"⚠️  Sample {idx}: No valid humans after filtering")
        return
    
    # Convert to pixel coordinates
    vis_boxes_px = xyxy_to_viz(visible_boxes[valid_indices], img_size)
    amod_boxes_px = xyxy_to_viz(amodal_boxes[valid_indices], img_size)
    occ_scores = occlusion_scores[valid_indices]
    
    # Get image info
    img_info = dataset.valid_images[idx]
    img_id = img_info['image_id']
    anns = dataset.img_to_anns[img_id]
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.imshow(img_np)
    ax.set_title(f'Sample {idx}: {num_humans} occluded human(s)', 
                 fontsize=18, fontweight='bold')
    ax.axis('off')
    
    # Draw boxes
    for i in range(num_humans):
        occ = occ_scores[i].item()
        
        # Color based on occlusion
        if occ < 0.2:
            vis_color, amod_color = 'lime', 'green'
        elif occ < 0.5:
            vis_color, amod_color = 'yellow', 'orange'
        else:
            vis_color, amod_color = 'red', 'darkred'
        
        # Amodal box (SOLID) - full extent
        x1, y1, x2, y2 = amod_boxes_px[i]
        w, h = x2 - x1, y2 - y1
        rect_amodal = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=3, edgecolor=amod_color, facecolor='none',
            linestyle='-', label='Amodal (full)' if i == 0 else ''
        )
        ax.add_patch(rect_amodal)
        
        # Visible box (DASHED) - visible part
        x1, y1, x2, y2 = vis_boxes_px[i]
        w, h = x2 - x1, y2 - y1
        rect_visible = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=vis_color, facecolor='none',
            linestyle='--', label='Visible' if i == 0 else ''
        )
        ax.add_patch(rect_visible)
        
        # Label
        category = anns[i]['category'] if i < len(anns) else 'human'
        label_text = f'{category}\nOcc: {occ:.2f}'
        
        # Position label above amodal box
        ax.text(amod_boxes_px[i][0], amod_boxes_px[i][1] - 15,
               label_text,
               color='white', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    # Legend
    ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
    
    # Info box
    info_text = (
        f"🟩 GREEN = Low occlusion (<20%)\n"
        f"🟨 YELLOW = Med occlusion (20-50%)\n"
        f"🟥 RED = High occlusion (>50%)\n\n"
        f"━━ SOLID = Amodal (full extent)\n"
        f"- - DASHED = Visible (seen part)\n\n"
        f"Amodal box ≥ Visible box"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'sample_{idx:04d}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Print info
    categories = [anns[i]['category'] for i in range(min(num_humans, len(anns)))]
    cat_str = ', '.join(categories[:3])
    if len(categories) > 3:
        cat_str += f', +{len(categories)-3} more'
    
    print(f"✅ Sample {idx:4d}: {num_humans:2d} humans [{cat_str}], "
          f"occ=[{occ_scores.min():.2f}, {occ_scores.max():.2f}]")


def main():
    parser = argparse.ArgumentParser(description='Visualize COCOA dataset')
    parser.add_argument('--image-dir', default='coco/train2014',
                        help='Path to COCO images')
    parser.add_argument('--ann-file', default='coco/COCO_amodal_train2014.json',
                        help='COCOA annotation file')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of samples to visualize')
    parser.add_argument('--min-occlusion', type=float, default=0.00,
                        help='Minimum occlusion rate')
    parser.add_argument('--output-dir', default='viz',
                        help='Output directory for visualizations')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔍 VISUALIZING COCOA DATASET - INDIVIDUAL OCCLUDED HUMANS")
    print("="*80)
    print(f"Annotation file: {args.ann_file}")
    print(f"Min occlusion: {args.min_occlusion:.2f}")
    print(f"Output: {args.output_dir}/")
    print("="*80 + "\n")
    
    # Load dataset
    dataset = COCOAAmodalDataset(
        image_dir=args.image_dir,
        ann_file=args.ann_file,
        split='train',
        image_size=640,
        max_objects=50,
        augment=False,
        min_occlusion=args.min_occlusion,
        min_area=400
    )
    
    if len(dataset) == 0:
        print("❌ No samples found! Check paths and min_occlusion threshold.")
        return
    
    print(f"\n🎨 Visualizing {min(args.num_samples, len(dataset))} samples...\n")
    
    # Select diverse samples
    samples_to_viz = []
    for idx in range(len(dataset)):
        img_info = dataset.valid_images[idx]
        img_id = img_info['image_id']
        anns = dataset.img_to_anns[img_id]
        
        if len(anns) > 0:
            max_occ = max(ann['occlude_rate'] for ann in anns)
            # Priority: more humans + higher occlusion
            priority = len(anns) * 10 + max_occ * 5
            samples_to_viz.append((priority, idx))
    
    samples_to_viz.sort(reverse=True)
    samples_to_viz = samples_to_viz[:args.num_samples]
    
    # Visualize
    for i, (priority, idx) in enumerate(samples_to_viz):
        print(f"[{i+1}/{len(samples_to_viz)}] ", end="")
        try:
            visualize_sample(dataset, idx, args.output_dir)
        except Exception as e:
            print(f"⚠️  Failed to visualize sample {idx}: {e}")
    
    print("\n" + "="*80)
    print(f"✅ Visualization complete!")
    print(f"📁 Check: {args.output_dir}/")
    print("="*80)
    print("\n💡 If the data looks good, start training:")
    print(f"   python amodal_detection_head/train_amodal.py \\")
    print(f"          --min-occlusion {args.min_occlusion} \\")
    print(f"          --batch-size 16 \\")
    print(f"          --epochs 50\n")


if __name__ == '__main__':
    main()