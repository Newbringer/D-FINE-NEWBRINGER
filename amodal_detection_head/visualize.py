#!/usr/bin/env python3
"""
Visualize COCOA Dataset - HUMANS ONLY
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'amodal_detection_head'))

from cocoa_dataset import COCOAAmodalDataset


def denormalize_image(image_tensor):
    """Denormalize image"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    return image


def box_cxcywh_to_xyxy(boxes, img_size):
    """Convert [cx, cy, w, h] normalized to [x1, y1, x2, y2] pixels"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    
    return torch.stack([x1, y1, x2, y2], dim=1)


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
    
    # Denormalize
    img_np = denormalize_image(image)
    img_size = image.shape[-1]
    
    # Get valid boxes
    valid_indices = torch.where(valid_mask > 0)[0]
    num_people = len(valid_indices)
    
    if num_people == 0:
        print(f"⚠️  Sample {idx}: No people!")
        return
    
    # Convert to pixels
    vis_boxes_px = box_cxcywh_to_xyxy(visible_boxes[valid_indices], img_size)
    amod_boxes_px = box_cxcywh_to_xyxy(amodal_boxes[valid_indices], img_size)
    occ_scores = occlusion_scores[valid_indices]
    
    # Get categories
    img_info = dataset.valid_images[idx]
    img_id = img_info['image_id']
    source = img_info.get('source', 'unknown')
    anns = dataset.img_to_anns[img_id]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.imshow(img_np)
    ax.set_title(f'Sample {idx}: {num_people} human(s) [Source: {source.upper()}]', 
                 fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Draw boxes
    for i in range(num_people):
        occ = occ_scores[i].item()
        
        # Colors based on occlusion
        if occ < 0.2:
            vis_color, amod_color = 'lime', 'green'
        elif occ < 0.5:
            vis_color, amod_color = 'yellow', 'orange'
        else:
            vis_color, amod_color = 'red', 'darkred'
        
        # Amodal (SOLID)
        x1, y1, x2, y2 = amod_boxes_px[i]
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=3, edgecolor=amod_color, facecolor='none',
            linestyle='-', label='Amodal' if i == 0 else ''
        )
        ax.add_patch(rect)
        
        # Visible (DASHED)
        x1, y1, x2, y2 = vis_boxes_px[i]
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=vis_color, facecolor='none',
            linestyle='--', label='Visible' if i == 0 else ''
        )
        ax.add_patch(rect)
        
        # Label
        category = anns[i]['name'] if i < len(anns) else 'person'
        ax.text(amod_boxes_px[i][0], amod_boxes_px[i][1] - 15,
               f'{category}\nOcc: {occ:.2f}',
               color='white', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.legend(loc='upper right', fontsize=12)
    
    # Info
    info_text = (
        f"🟩 GREEN = Low occ (< 20%)\n"
        f"🟨 YELLOW = Med occ (20-50%)\n"
        f"🟥 RED = High occ (> 50%)\n\n"
        f"SOLID = Amodal (full)\n"
        f"DASHED = Visible\n\n"
        f"Source: {source.upper()}"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'sample_{idx:03d}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    categories = [anns[i]['name'] for i in range(min(num_people, len(anns)))]
    print(f"✅ Sample {idx}: {num_people} humans [{', '.join(categories[:3])}], occ=[{occ_scores.min():.2f}, {occ_scores.max():.2f}]")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', default='coco/train2014')
    parser.add_argument('--official-ann', default='coco/COCO_amodal_train2014.json')
    parser.add_argument('--detectron-ann', default='coco/COCO_amodal_train2014_detectron.json')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--min-occlusion', type=float, default=0.0)
    parser.add_argument('--output-dir', default='viz')
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 VISUALIZING COCOA - HUMANS ONLY")
    print("="*80)
    print(f"📂 Official: {args.official_ann}")
    print(f"📂 Detectron backup: {args.detectron_ann}")
    print(f"📊 Min occlusion: {args.min_occlusion:.2f}")
    print("="*80)
    
    # Load dataset
    print("\n📦 Loading dataset...")
    dataset = COCOAAmodalDataset(
        image_dir=args.image_dir,
        official_ann_file=args.official_ann,
        detectron_ann_file=args.detectron_ann,
        split='train',
        image_size=640,
        max_objects=50,
        augment=False,
        min_occlusion=args.min_occlusion,
        min_area=400
    )
    
    if len(dataset) == 0:
        print("❌ No samples found!")
        return
    
    print(f"\n🎨 Visualizing {min(args.num_samples, len(dataset))} samples...")
    
    # Find interesting samples
    samples = []
    for idx in range(len(dataset)):
        img_info = dataset.valid_images[idx]
        img_id = img_info['image_id']
        source = img_info.get('source', 'unknown')
        anns = dataset.img_to_anns[img_id]
        
        if len(anns) > 0:
            max_occ = max(ann['occlude_rate'] for ann in anns)
            # Boost detectron priority to show both sources
            priority = len(anns) * 10 + max_occ * 5
            if source == 'detectron':
                priority += 2
            samples.append((priority, idx, source))
    
    samples.sort(reverse=True)
    samples = samples[:args.num_samples]
    
    print(f"   Official samples: {sum(1 for _, _, s in samples if s == 'official')}")
    print(f"   Detectron samples: {sum(1 for _, _, s in samples if s == 'detectron')}")
    
    for i, (_, idx, source) in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] ", end="")
        visualize_sample(dataset, idx, args.output_dir)
    
    print("\n" + "="*80)
    print(f"✅ Complete! Check: {args.output_dir}/")
    print("="*80)
    
    print("\n🚀 If data looks good, train:")
    print(f"   python amodal_detection_head/train_amodal.py --min-occlusion {args.min_occlusion}")


if __name__ == '__main__':
    main()