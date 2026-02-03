#!/usr/bin/env python3
"""
Test DFINE detection only - no amodal head
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'segmentation_sivert'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))


def load_dfine_model(config_path, checkpoint_path, device):
    """Load DFINE model"""
    try:
        from core.models import load_pretrained_dfine
        model = load_pretrained_dfine(config_path, checkpoint_path)
    except:
        try:
            from src.core import YAMLConfig
        except:
            from core import YAMLConfig
        
        cfg = YAMLConfig(str(config_path))
        model = cfg.model
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'ema' in checkpoint and 'module' in checkpoint['ema']:
            state_dict = checkpoint['ema']['module']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
    
    return model.to(device)


def postprocess_dfine_outputs(outputs, orig_size, conf_threshold, device, num_top_queries=300):
    """Postprocess DFINE outputs"""
    orig_w, orig_h = orig_size
    
    # Extract raw predictions
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4] in cxcywh normalized
    
    num_classes = pred_logits.shape[-1]
    
    print(f"   pred_logits: {pred_logits.shape}")
    print(f"   pred_boxes: {pred_boxes.shape}")
    
    # Convert boxes from cxcywh (normalized) to xyxy (pixel coords)
    cx, cy, w, h = pred_boxes.unbind(dim=-1)
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    
    # Apply sigmoid to logits
    scores = pred_logits.sigmoid()
    
    print(f"   scores range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   scores > {conf_threshold}: {(scores > conf_threshold).sum().item()}")
    
    # DFINE-style top-k selection across all queries and classes
    scores_flat = scores.flatten()
    topk_values, topk_indices = torch.topk(
        scores_flat,
        k=min(num_top_queries, scores_flat.numel())
    )
    
    # Convert flat indices back to (query_idx, class_idx)
    labels = torch.remainder(topk_indices, num_classes)
    query_indices = topk_indices // num_classes
    boxes = boxes_xyxy[query_indices]
    scores = topk_values
    
    # Filter by confidence threshold
    keep = scores > conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    print(f"   After confidence filter: {len(boxes)} detections")
    
    # Show what we have before NMS
    if len(boxes) > 0:
        print(f"\n   Detections before NMS:")
        for i in range(min(10, len(boxes))):
            print(f"      Class {labels[i].item()}: {scores[i].item():.3f}")
    
    # Apply NMS per class
    if len(boxes) > 0:
        from torchvision.ops import nms
        keep_indices = []
        for class_id in torch.unique(labels):
            class_mask = labels == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = torch.where(class_mask)[0]
            
            nms_keep = nms(class_boxes.float(), class_scores.float(), 0.65)
            keep_indices.append(class_indices[nms_keep])
        
        if keep_indices:
            keep_indices = torch.cat(keep_indices)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
    
    return boxes, scores, labels


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def visualize_detections(image, boxes, scores, labels, output_path):
    """Visualize DFINE detections"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    ax.imshow(image)
    ax.set_title(f'DFINE Detections ({len(boxes)} objects)', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        class_idx = int(label)
        class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
        
        # Color based on class
        color = 'lime' if class_idx == 0 else 'yellow'
        
        rect = patches.Rectangle((x1, y1), w, h, linewidth=3,
                                edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        ax.text(x1, y1-10, f'{class_name} {score:.2f}',
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--dfine-config', default='models/dfine_hgnetv2_x_obj2coco.yml')
    parser.add_argument('--dfine-checkpoint', default='models/dfine_0.73.pth')
    parser.add_argument('--output', default='dfine_result.png')
    parser.add_argument('--conf-threshold', type=float, default=0.3)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print("🔍 DFINE DETECTION ONLY")
    print(f"{'='*80}\n")
    
    # Load model
    print("📦 Loading DFINE...")
    dfine_model = load_dfine_model(args.dfine_config, args.dfine_checkpoint, device)
    dfine_model.eval()
    print("   ✅ DFINE loaded\n")
    
    # Load image
    print(f"🖼️  Loading image: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Could not load image: {args.image}")
    
    orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_img.shape[:2]
    
    # Resize to 640x640
    img_resized = cv2.resize(orig_img, (640, 640))
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - mean) / std
    
    # To tensor
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    print(f"   Size: {orig_w}x{orig_h} → 640x640\n")
    
    # Run DFINE
    print("🔍 Running DFINE detection...")
    with torch.no_grad():
        outputs = dfine_model(img_tensor)
    
    # Postprocess
    boxes, scores, labels = postprocess_dfine_outputs(
        outputs, (orig_w, orig_h), args.conf_threshold, device
    )
    
    print(f"\n   ✅ Final: {len(boxes)} detections\n")
    
    if len(boxes) == 0:
        print("❌ No objects detected!")
        print("   Try lowering --conf-threshold\n")
        return
    
    # Print detections
    print("📊 Detections:")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        class_idx = int(label)
        class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
        print(f"   {i+1}. {class_name}: {score:.3f}")
    print()
    
    # Visualize
    print("🎨 Creating visualization...")
    visualize_detections(
        img_resized,
        boxes.cpu().numpy(),
        scores.cpu().numpy(),
        labels.cpu().numpy(),
        args.output
    )
    
    print(f"{'='*80}")
    print("✅ Complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()