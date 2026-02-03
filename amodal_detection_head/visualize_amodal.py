#!/usr/bin/env python3
"""
Visualize Amodal Bounding Box Predictions
Shows both visible detections and predicted amodal boxes
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
sys.path.insert(0, str(PROJECT_ROOT / 'amodal_detection_head'))
sys.path.insert(0, str(PROJECT_ROOT / 'glass_wall_detection' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'segmentation_sivert'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

from amodal_head import AmodalOffsetHead, extract_roi_features


def load_amodal_model(config_path, checkpoint_path, device):
    """Load full amodal model (DFINE + amodal head)"""
    print(f"\n📦 Loading amodal model from: {checkpoint_path}")
    
    # Import architecture
    from model_architecture import SegmentationHead, DFineWithSegmentation
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check for required keys
    if 'dfine_model' not in checkpoint or 'amodal_head' not in checkpoint:
        raise ValueError(
            f"Checkpoint must contain both 'dfine_model' and 'amodal_head' keys!\n"
            f"Found keys: {list(checkpoint.keys())}"
        )
    
    dfine_state = checkpoint['dfine_model']
    amodal_state = checkpoint['amodal_head']
    
    print(f"   ✅ Found DFINE and amodal head in checkpoint")
    print(f"   Best GIoU: {checkpoint.get('best_giou', 'N/A')}")
    print(f"   Best IoU: {checkpoint.get('best_iou', 'N/A')}")
    
    # Load base DFINE config
    try:
        from src.core import YAMLConfig
    except:
        from core import YAMLConfig
    
    cfg = YAMLConfig(str(config_path))
    base_model = cfg.model
    
    # Get backbone channels
    print("   Analyzing backbone...")
    base_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640)
        backbone_features = base_model.backbone(dummy_input)
        backbone_channels = [feat.shape[1] for feat in backbone_features]
    
    print(f"   Backbone channels: {backbone_channels}")
    
    # Infer hyperparameters from checkpoint
    feature_dim = 256
    for key in dfine_state.keys():
        if 'seg_head.fpn.lateral_convs.0.weight' in key:
            feature_dim = dfine_state[key].shape[0]
            break
    
    print(f"   Feature dim: {feature_dim}")
    
    # Create segmentation head
    seg_head = SegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=7,
        feature_dim=feature_dim,
        dropout_rate=0.1
    )
    
    # Create combined DFINE model
    dfine_model = DFineWithSegmentation(
        dfine_model=base_model,
        seg_head=seg_head,
        freeze_detection=False
    )
    
    # Load DFINE weights
    dfine_model.load_state_dict(dfine_state, strict=False)
    
    # Create amodal head
    args = checkpoint.get('args', {})
    amodal_head = AmodalOffsetHead(
        in_channels=backbone_channels[-1],
        hidden_dim=args.get('hidden_dim', 512),
        roi_size=args.get('roi_size', 7)
    )
    
    # Load amodal weights
    amodal_head.load_state_dict(amodal_state)
    
    print(f"   ✅ Loaded both models successfully")
    
    return dfine_model.to(device), amodal_head.to(device)


def detect_with_dfine(model, image_tensor, conf_threshold=0.3, nms_threshold=0.65, num_top_queries=300):
    """Run DFINE detection and return boxes, scores, labels"""
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Extract predictions
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4] cxcywh normalized
    
    num_classes = pred_logits.shape[-1]
    
    # Apply sigmoid
    scores = pred_logits.sigmoid()
    
    # Top-k selection
    scores_flat = scores.flatten()
    topk_values, topk_indices = torch.topk(
        scores_flat,
        k=min(num_top_queries, scores_flat.numel())
    )
    
    # Convert to query and class indices
    labels = torch.remainder(topk_indices, num_classes)
    query_indices = topk_indices // num_classes
    boxes_cxcywh = pred_boxes[query_indices]
    scores = topk_values
    
    # Filter by confidence
    keep = scores > conf_threshold
    boxes_cxcywh = boxes_cxcywh[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Apply NMS per class
    if len(boxes_cxcywh) > 0:
        from torchvision.ops import nms, box_convert
        
        # Convert to xyxy for NMS
        boxes_xyxy = box_convert(boxes_cxcywh, 'cxcywh', 'xyxy')
        
        keep_indices = []
        for class_id in torch.unique(labels):
            class_mask = labels == class_id
            class_boxes = boxes_xyxy[class_mask]
            class_scores = scores[class_mask]
            class_indices = torch.where(class_mask)[0]
            
            nms_keep = nms(class_boxes.float(), class_scores.float(), nms_threshold)
            keep_indices.append(class_indices[nms_keep])
        
        if keep_indices:
            keep_indices = torch.cat(keep_indices)
            boxes_cxcywh = boxes_cxcywh[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
    
    return boxes_cxcywh, scores, labels


def predict_amodal_boxes(amodal_head, dfine_model, visible_boxes_norm, feature_map, roi_size):
    """Predict amodal boxes from visible boxes"""
    if len(visible_boxes_norm) == 0:
        return visible_boxes_norm.clone()
    
    # Extract RoI features
    roi_features = extract_roi_features(feature_map, visible_boxes_norm, roi_size)
    
    # Predict amodal
    with torch.no_grad():
        predictions = amodal_head(roi_features, visible_boxes_norm)
    
    return predictions['amodal_boxes']


def normalize_boxes(boxes_cxcywh, image_size):
    """Convert cxcywh boxes to normalized xyxy [0,1]"""
    from torchvision.ops import box_convert
    
    # Convert to xyxy
    boxes_xyxy = box_convert(boxes_cxcywh, 'cxcywh', 'xyxy')
    
    # Already normalized [0, 1]
    boxes_xyxy = boxes_xyxy.clamp(0, 1)
    
    return boxes_xyxy


def denormalize_boxes(boxes_norm, orig_w, orig_h):
    """Convert normalized xyxy [0,1] to pixel coords"""
    boxes_px = boxes_norm.clone()
    boxes_px[:, [0, 2]] *= orig_w
    boxes_px[:, [1, 3]] *= orig_h
    return boxes_px


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


def visualize_amodal_predictions(image, visible_boxes, amodal_boxes, scores, labels, 
                                output_path, title, show_only_person=True):
    """Visualize visible and amodal boxes
    
    Args:
        image: RGB image
        visible_boxes: [N, 4] xyxy in pixel coords
        amodal_boxes: [N, 4] xyxy in pixel coords
        scores: [N] confidence scores
        labels: [N] class labels
        show_only_person: Only show person detections
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    
    ax.imshow(image)
    ax.axis('off')
    
    # Filter for person only if requested
    if show_only_person:
        person_mask = labels == 0
        visible_boxes = visible_boxes[person_mask]
        amodal_boxes = amodal_boxes[person_mask]
        scores = scores[person_mask]
        labels = labels[person_mask]
    
    num_detections = len(visible_boxes)
    
    ax.set_title(f'{title}\n{num_detections} {"person" if show_only_person else "object"} detections',
                 fontsize=16, fontweight='bold')
    
    # Draw boxes
    for i, (vis_box, amod_box, score, label) in enumerate(zip(visible_boxes, amodal_boxes, scores, labels)):
        class_idx = int(label)
        class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
        
        # Compute occlusion estimate
        vis_area = (vis_box[2] - vis_box[0]) * (vis_box[3] - vis_box[1])
        amod_area = (amod_box[2] - amod_box[0]) * (amod_box[3] - amod_box[1])
        occlusion_est = max(0, 1 - vis_area / (amod_area + 1e-6))
        
        # Color based on estimated occlusion
        if occlusion_est < 0.1:
            vis_color, amod_color = 'lime', 'green'
        elif occlusion_est < 0.3:
            vis_color, amod_color = 'yellow', 'orange'
        else:
            vis_color, amod_color = 'red', 'darkred'
        
        # Draw AMODAL box (SOLID) - full extent prediction
        x1, y1, x2, y2 = amod_box
        w, h = x2 - x1, y2 - y1
        rect_amodal = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=3, edgecolor=amod_color, facecolor='none',
            linestyle='-', label='Amodal (predicted)' if i == 0 else ''
        )
        ax.add_patch(rect_amodal)
        
        # Draw VISIBLE box (DASHED) - detected part
        x1, y1, x2, y2 = vis_box
        w, h = x2 - x1, y2 - y1
        rect_visible = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=vis_color, facecolor='none',
            linestyle='--', label='Visible (detected)' if i == 0 else ''
        )
        ax.add_patch(rect_visible)
        
        # Label
        label_text = f'{class_name} {score:.2f}\nOcc: ~{occlusion_est:.2f}'
        
        # Position label above amodal box
        ax.text(amod_box[0], amod_box[1] - 15,
               label_text,
               color='white', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    # Legend
    if num_detections > 0:
        ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
    
    # Info box
    info_text = (
        f"━━ SOLID = Amodal (full extent)\n"
        f"- - DASHED = Visible (detected)\n\n"
        f"🟩 GREEN = Low occlusion\n"
        f"🟨 YELLOW = Med occlusion\n"
        f"🟥 RED = High occlusion"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Amodal Predictions')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--config', default='models/dfine_hgnetv2_x_obj2coco.yml',
                        help='DFINE config file')
    parser.add_argument('--checkpoint', default='outputs/amodal_humans/best_model.pth',
                        help='Amodal checkpoint')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                        help='Detection confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.65,
                        help='NMS threshold')
    parser.add_argument('--output-dir', default='amodal_viz',
                        help='Output directory')
    parser.add_argument('--show-all', action='store_true',
                        help='Show all classes (not just person)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print("🎯 AMODAL BOUNDING BOX VISUALIZATION")
    print(f"{'='*80}")
    print(f"Image: {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    dfine_model, amodal_head = load_amodal_model(args.config, args.checkpoint, device)
    dfine_model.eval()
    amodal_head.eval()
    
    # Get args from checkpoint for ROI size
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    roi_size = checkpoint.get('args', {}).get('roi_size', 7)
    
    # Load and preprocess image
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
    
    print(f"   Image size: {orig_w}x{orig_h} → 640x640")
    
    # Run detection
    print(f"\n🔍 Running DFINE detection...")
    boxes_cxcywh, scores, labels = detect_with_dfine(
        dfine_model, img_tensor, args.conf_threshold, args.nms_threshold
    )
    
    print(f"   Detected {len(boxes_cxcywh)} objects")
    
    if len(boxes_cxcywh) == 0:
        print(f"❌ No detections found! Try lowering --conf-threshold")
        return
    
    # Count by class
    class_counts = {}
    for label in labels:
        class_idx = int(label)
        class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"   Detections by class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"      - {class_name}: {count}")
    
    # Normalize boxes for amodal head
    visible_boxes_norm = normalize_boxes(boxes_cxcywh, 640)
    
    # Extract features for amodal prediction
    print(f"\n🎨 Predicting amodal boxes...")
    with torch.no_grad():
        # Get backbone features
        if hasattr(dfine_model, 'dfine_model'):
            features = dfine_model.dfine_model.backbone(img_tensor)
        else:
            features = dfine_model.backbone(img_tensor)
        
        if isinstance(features, (list, tuple)):
            feature_map = features[-1]
        else:
            feature_map = features
    
    # Predict amodal boxes
    amodal_boxes_norm = predict_amodal_boxes(
        amodal_head, dfine_model, visible_boxes_norm, feature_map, roi_size
    )
    
    # Denormalize boxes to original image size
    visible_boxes_px = denormalize_boxes(visible_boxes_norm, 640, 640)
    amodal_boxes_px = denormalize_boxes(amodal_boxes_norm, 640, 640)
    
    # Visualize
    print(f"\n🎨 Creating visualization...")
    output_path = os.path.join(args.output_dir, 'amodal_predictions.png')
    
    visualize_amodal_predictions(
        img_resized,
        visible_boxes_px.detach().cpu().numpy(),
        amodal_boxes_px.detach().cpu().numpy(),
        scores.detach().cpu().numpy(),
        labels.detach().cpu().numpy(),
        output_path,
        'Amodal Detection Predictions',
        show_only_person=not args.show_all
    )
    
    # Print statistics
    print(f"\n📊 Statistics:")
    person_mask = labels == 0
    if person_mask.sum() > 0:
        vis_areas = ((visible_boxes_px[:, 2] - visible_boxes_px[:, 0]) * 
                     (visible_boxes_px[:, 3] - visible_boxes_px[:, 1]))
        amod_areas = ((amodal_boxes_px[:, 2] - amodal_boxes_px[:, 0]) * 
                      (amodal_boxes_px[:, 3] - amodal_boxes_px[:, 1]))
        
        person_vis = vis_areas[person_mask]
        person_amod = amod_areas[person_mask]
        
        expansions = (person_amod / (person_vis + 1e-6) - 1) * 100
        
        print(f"   Person detections: {person_mask.sum()}")
        print(f"   Avg expansion: {expansions.mean():.1f}%")
        print(f"   Max expansion: {expansions.max():.1f}%")
        print(f"   Min expansion: {expansions.min():.1f}%")
    
    print(f"\n{'='*80}")
    print(f"✅ Visualization complete!")
    print(f"📁 Output: {output_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()