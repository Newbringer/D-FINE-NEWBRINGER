#!/usr/bin/env python3
"""
Quick script to visualize annotations and check if boxes cover full panes
"""
import cv2
import json
import numpy as np
from pathlib import Path
import sys

def check_annotation_sizes(data_dir):
    """Check if bounding boxes seem reasonable for full panes"""
    data_dir = Path(data_dir)
    
    # Load annotations
    ann_file = data_dir / '_annotations.coco.json'
    if not ann_file.exists():
        print(f"❌ Annotations not found: {ann_file}")
        return
    
    with open(ann_file) as f:
        coco_data = json.load(f)
    
    # Analyze box sizes
    box_areas = []
    box_aspect_ratios = []
    
    for ann in coco_data['annotations']:
        x, y, w, h = ann['bbox']
        area_ratio = (w * h) / (640 * 640)  # Assuming 640x640 images
        aspect_ratio = h / w if w > 0 else 0
        
        box_areas.append(area_ratio)
        box_aspect_ratios.append(aspect_ratio)
    
    # Stats
    avg_area = np.mean(box_areas)
    min_area = np.min(box_areas)
    max_area = np.max(box_areas)
    
    print("📊 Annotation Statistics")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"\n📏 Bounding Box Sizes (relative to image):")
    print(f"  Average: {avg_area:.1%} of image")
    print(f"  Min: {min_area:.1%}")
    print(f"  Max: {max_area:.1%}")
    
    # Check if boxes seem too small
    too_small = sum(1 for a in box_areas if a < 0.05)  # Less than 5% of image
    if too_small > 0:
        print(f"\n⚠️  WARNING: {too_small} boxes are very small (< 5% of image)")
        print(f"   → These might only cover text, not full panes!")
    
    print(f"\n📐 Aspect Ratios (height/width):")
    print(f"  Average: {np.mean(box_aspect_ratios):.2f}")
    print(f"  Range: {np.min(box_aspect_ratios):.2f} - {np.max(box_aspect_ratios):.2f}")
    
    if np.mean(box_aspect_ratios) > 2.0:
        print(f"   → Panes are tall and narrow (typical for glass walls) ✅")
    elif np.mean(box_aspect_ratios) < 0.5:
        print(f"   → Panes are wide and short")
    else:
        print(f"   → Panes are roughly square")

def visualize_samples(data_dir, num_samples=5):
    """Visualize a few annotated samples"""
    data_dir = Path(data_dir)
    ann_file = data_dir / '_annotations.coco.json'
    
    with open(ann_file) as f:
        coco_data = json.load(f)
    
    # Create image ID to annotations mapping
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    print(f"\n🖼️  Visualizing {num_samples} sample annotations...\n")
    
    output_dir = Path('annotation_check_samples')
    output_dir.mkdir(exist_ok=True)
    
    for i, img_info in enumerate(coco_data['images'][:num_samples]):
        img_path = data_dir / img_info['file_name']
        if not img_path.exists():
            continue
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Draw annotations
        anns = img_to_anns.get(img_info['id'], [])
        for ann in anns:
            x, y, w, h = ann['bbox']
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Calculate box size
            img_h, img_w = img.shape[:2]
            size_pct = (w * h) / (img_w * img_h) * 100
            
            # Add label with size
            label = f"Pane ({size_pct:.1f}% of image)"
            cv2.putText(img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save
        output_path = output_dir / f"sample_{i+1}.jpg"
        cv2.imwrite(str(output_path), img)
        print(f"  ✅ Saved: {output_path} ({len(anns)} panes)")
    
    print(f"\n📁 Check the '{output_dir}/' folder to verify annotations!")
    print(f"   → Boxes should cover ENTIRE panes, not just text")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_annotations.py <data_directory>")
        print("Example: python check_annotations.py ../data/glass_wall/train")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    check_annotation_sizes(data_dir)
    visualize_samples(data_dir, num_samples=5)
