#!/usr/bin/env python3
"""
Visualize COCOA Ground Truth Annotations
Shows both visible and amodal bounding boxes for humans (man, woman, boy, girl, etc.)
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


def visualize_ground_truth(dataset, idx, output_dir='data_check'):
    """
    Visualize ground truth annotations with both visible and amodal boxes
    
    Args:
        dataset: COCOA dataset
        idx: Sample index
        output_dir: Directory to save visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data
    sample = dataset[idx]
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
    
    # Get actual object info from dataset
    img_info = dataset.valid_images[idx]
    img_id = img_info['image_id']
    anns = dataset.img_to_anns[img_id]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.imshow(img_np)
    ax.set_title(f'Sample {idx}: {num_people} human(s) - Amodal Detection GT\n{Path(file_name).name}', 
                 fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Color map for different occlusion levels
    def get_color_for_occlusion(occ):
        """Get color based on occlusion level"""
        if occ < 0.2:
            return 'lime', 'green'  # Low occlusion
        elif occ < 0.5:
            return 'yellow', 'orange'  # Medium occlusion
        else:
            return 'red', 'darkred'  # High occlusion
    
    # Draw boxes
    for i in range(num_people):
        occ = occ_scores[i].item()
        vis_color, amod_color = get_color_for_occlusion(occ)
        
        # Amodal box (SOLID, BRIGHTER) - Full human extent
        x1, y1, x2, y2 = amod_boxes_px[i]
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=3, edgecolor=amod_color, facecolor='none',
            linestyle='-', label='Amodal (full)' if i == 0 else ''
        )
        ax.add_patch(rect)
        
        # Visible box (DASHED, DIMMER) - What you can see
        x1, y1, x2, y2 = vis_boxes_px[i]
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=vis_color, facecolor='none',
            linestyle='--', label='Visible' if i == 0 else ''
        )
        ax.add_patch(rect)
        
        # Get category name from annotation
        category = anns[i]['name'] if i < len(anns) else 'human'
        
        # Show label with category and occlusion
        label_text = f'{category}\nOcc: {occ:.2f}'
        
        # Position label above amodal box
        ax.text(amod_boxes_px[i][0], amod_boxes_px[i][1] - 15, label_text,
               color='white', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.legend(loc='upper right', fontsize=12)
    
    # Add interpretation text
    info_text = (
        f"🟩 GREEN = Low occlusion (< 20%)\n"
        f"🟨 YELLOW = Medium occlusion (20-50%)\n"
        f"🟥 RED = High occlusion (> 50%)\n\n"
        f"SOLID line = Amodal (full extent)\n"
        f"DASHED line = Visible (what you see)"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'gt_sample_{idx:03d}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Print info
    categories = [anns[i]['name'] for i in range(min(num_people, len(anns)))]
    print(f"✅ Sample {idx}: {num_people} human(s) [{', '.join(categories[:5])}]")
    print(f"   Occlusion range: [{occ_scores.min():.2f}, {occ_scores.max():.2f}]")
    print(f"   Saved: {output_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize COCOA Ground Truth')
    parser.add_argument('--image-dir', default='coco/train2014',
                        help='Path to COCO images')
    parser.add_argument('--annotation-file', default='coco/COCO_amodal_train2014.json',
                        help='Path to COCOA annotation JSON')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples to visualize')
    parser.add_argument('--output-dir', default='data_check_humans',
                        help='Output directory for visualizations')
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 VISUALIZING COCOA HUMAN ANNOTATIONS (ALL CATEGORIES)")
    print("="*80)
    print(f"📂 Images: {args.image_dir}")
    print(f"📂 Annotations: {args.annotation_file}")
    print(f"📊 Samples to visualize: {args.num_samples}")
    print(f"👥 Categories: man, woman, boy, girl, people, person, child, etc.")
    print("="*80)
    
    # Load dataset
    print("\n📦 Loading COCOA dataset...")
    dataset = COCOAAmodalDataset(
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        split='train',
        image_size=640,
        max_objects=50,
        augment=False,
        person_only=True  # This now includes all human categories
    )
    
    if len(dataset) == 0:
        print("❌ No samples found in dataset!")
        return
    
    print(f"\n✅ Dataset loaded: {len(dataset)} images with humans")
    print(f"\n🎨 Creating visualizations...")
    print("-"*80)
    
    # Visualize samples
    num_samples = min(args.num_samples, len(dataset))
    
    # Try to get samples with varying occlusion and multiple people
    samples_to_viz = []
    
    # First, scan dataset to find interesting samples
    print(f"\n🔍 Scanning dataset for interesting samples...")
    for idx in range(len(dataset)):
        sample = dataset[idx]
        num_valid = sample['valid_mask'].sum().item()
        
        if num_valid > 0:
            occ_scores = sample['occlusion_scores'][sample['valid_mask'] > 0]
            max_occ = occ_scores.max().item()
            
            # Prioritize samples with:
            # - Multiple people
            # - High occlusion
            # - Variety
            priority = num_valid * 10 + max_occ * 5
            
            samples_to_viz.append((priority, idx, num_valid, max_occ))
    
    # Sort by priority and take top samples
    samples_to_viz.sort(reverse=True)
    samples_to_viz = samples_to_viz[:num_samples]
    
    print(f"   Selected {len(samples_to_viz)} interesting samples")
    print(f"   Max people in a sample: {max(s[2] for s in samples_to_viz):.0f}")
    print(f"   Max occlusion: {max(s[3] for s in samples_to_viz):.2f}")
    
    # Visualize
    print(f"\n📸 Generating visualizations...")
    for i, (priority, idx, num_people, max_occ) in enumerate(samples_to_viz):
        print(f"\n[{i+1}/{len(samples_to_viz)}] ", end="")
        visualize_ground_truth(dataset, idx, args.output_dir)
    
    print("\n" + "="*80)
    print(f"✅ Visualization complete!")
    print(f"📁 Check images in: {args.output_dir}/")
    print("="*80)
    
    print("\n💡 What you should see:")
    print("   ✅ SOLID colored boxes = Full human extent (amodal)")
    print("   ✅ DASHED boxes = Visible parts only")
    print("   ✅ Different colors for occlusion levels")
    print("   ✅ Labels showing category (man/woman/boy/girl) + occlusion")
    print("\n   ❌ If boxes look wrong, there's a data loading issue")
    print("   ❌ If all boxes are identical, visible bbox computation is off")
    print("\n🚀 If visualizations look good, you're ready to train!")
    print("   python amodal_detection_head/train_amodal.py --seg-checkpoint path/to/checkpoint.pth")


if __name__ == '__main__':
    main()