#!/usr/bin/env python3
"""
Test DFINE detection - verify baseline and amodal-trained models
Tests detection capabilities before amodal head visualization
FIXED: Properly loads segmentation model architecture
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
sys.path.insert(0, str(PROJECT_ROOT / 'glass_wall_detection' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'segmentation_sivert'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))


def load_segmentation_model(config_path, state_dict, device):
    """Load DFINE segmentation model (used by dfine_0.73.pth and amodal checkpoints)"""
    # Import architecture components
    from model_architecture import SegmentationHead, DFineWithSegmentation
    
    # Load base DFINE config
    try:
        from src.core import YAMLConfig
    except:
        from core import YAMLConfig
    
    print(f"   Loading config: {config_path}")
    cfg = YAMLConfig(str(config_path))
    base_model = cfg.model
    
    # Get backbone channels
    base_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640)
        backbone_features = base_model.backbone(dummy_input)
        backbone_channels = [feat.shape[1] for feat in backbone_features]
    
    print(f"   Backbone channels: {backbone_channels}")
    
    # Infer feature_dim from checkpoint
    feature_dim = 256  # default
    for key in state_dict.keys():
        if 'seg_head.fpn.lateral_convs.0.weight' in key:
            feature_dim = state_dict[key].shape[0]
            break
        elif 'seg_head.decoder.0.weight' in key:
            feature_dim = state_dict[key].shape[1]
            break
    
    print(f"   Feature dim: {feature_dim}")
    
    # Create segmentation head
    seg_head = SegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=7,  # Pascal Person Parts
        feature_dim=feature_dim,
        dropout_rate=0.1
    )
    
    # Create combined model
    model = DFineWithSegmentation(
        dfine_model=base_model,
        seg_head=seg_head,
        freeze_detection=False  # Not training
    )
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"   ⚠️  Missing keys: {len(missing)}")
    if unexpected:
        print(f"   ⚠️  Unexpected keys: {len(unexpected)}")
    
    # For testing, we only need the DFINE part
    model = model.dfine_model
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ DFINE model extracted - {total_params:,} parameters")
    
    return model


def load_dfine_model(config_path, checkpoint_path, device):
    """Load DFINE model, handling standard, segmentation, and amodal checkpoints"""
    print(f"\n📦 Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if 'ema' in checkpoint and 'module' in checkpoint['ema']:
        state_dict = checkpoint['ema']['module']
        print("   Using EMA weights")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("   Using model_state_dict")
    elif 'dfine_model' in checkpoint:
        # This is an amodal checkpoint with separate dfine_model key
        state_dict = checkpoint['dfine_model']
        print("   Using dfine_model from amodal checkpoint")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("   Using model weights")
    else:
        state_dict = checkpoint
        print("   Using checkpoint directly")
    
    # Check what type of checkpoint this is
    has_seg_head = any('seg_head.' in key for key in state_dict.keys())
    has_dfine_prefix = any('dfine_model.' in key for key in state_dict.keys())
    has_amodal = any('amodal_head.' in key for key in state_dict.keys())
    
    print(f"   Checkpoint type: seg_head={has_seg_head}, dfine_prefix={has_dfine_prefix}, amodal={has_amodal}")
    
    # If this is a DFineWithSegmentation checkpoint, we need to recreate the architecture
    if has_seg_head or has_dfine_prefix:
        print("   Detected segmentation model - recreating architecture...")
        return load_segmentation_model(config_path, state_dict, device)
    
    # Otherwise, load as standard DFINE
    try:
        from src.core import YAMLConfig
    except:
        from core import YAMLConfig
    
    print(f"   Loading config: {config_path}")
    cfg = YAMLConfig(str(config_path))
    model = cfg.model
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"   ⚠️  Missing keys: {len(missing)}")
        if len(missing) <= 5:
            for key in missing[:5]:
                print(f"      - {key}")
    if unexpected:
        print(f"   ⚠️  Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 5:
            for key in unexpected[:5]:
                print(f"      - {key}")
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model loaded - {total_params:,} parameters")
    
    return model


def postprocess_dfine_outputs(outputs, orig_size, conf_threshold, device, num_top_queries=300):
    """Postprocess DFINE outputs - using proven approach from glass_wall detection"""
    orig_w, orig_h = orig_size
    
    # Extract raw predictions
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4] in cxcywh normalized
    
    num_classes = pred_logits.shape[-1]
    
    print(f"\n   Raw outputs:")
    print(f"   - pred_logits: {pred_logits.shape}")
    print(f"   - pred_boxes: {pred_boxes.shape}")
    print(f"   - num_classes: {num_classes}")
    
    # Convert boxes from cxcywh (normalized) to xyxy (pixel coords)
    cx, cy, w, h = pred_boxes.unbind(dim=-1)
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    
    # Apply sigmoid to logits
    scores = pred_logits.sigmoid()
    
    print(f"   - scores range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   - scores > {conf_threshold}: {(scores > conf_threshold).sum().item()}")
    
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
    
    print(f"   - After top-k: {len(boxes)} candidates")
    
    # Filter by confidence threshold
    keep = scores > conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    print(f"   - After conf filter: {len(boxes)} detections")
    
    # Show what we have before NMS
    if len(boxes) > 0:
        print(f"\n   Top detections before NMS:")
        for i in range(min(5, len(boxes))):
            class_idx = int(labels[i])
            class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
            print(f"      {i+1}. {class_name}: {scores[i].item():.3f}")
    
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
    
    print(f"   - After NMS: {len(boxes)} final detections")
    
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


def visualize_detections(image, boxes, scores, labels, output_path, title):
    """Visualize DFINE detections"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    
    ax.imshow(image)
    ax.set_title(f'{title}\n{len(boxes)} detections', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Count by class
    class_counts = {}
    for label in labels:
        class_idx = int(label)
        class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        class_idx = int(label)
        class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
        
        # Color: green for person, yellow for others
        color = 'lime' if class_idx == 0 else 'yellow'
        
        rect = patches.Rectangle((x1, y1), w, h, linewidth=3,
                                edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        ax.text(x1, y1-10, f'{class_name} {score:.2f}',
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    # Add summary
    summary = '\n'.join([f'{k}: {v}' for k, v in sorted(class_counts.items())])
    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved visualization to: {output_path}")


def test_model(model, image_path, conf_threshold, device, output_path, model_name):
    """Test a single model on an image"""
    print(f"\n{'='*80}")
    print(f"🔍 Testing: {model_name}")
    print(f"{'='*80}")
    
    # Load image
    print(f"\n🖼️  Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
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
    
    print(f"   Size: {orig_w}x{orig_h} → 640x640")
    
    # Run inference
    print(f"\n🔍 Running inference...")
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Postprocess
    boxes, scores, labels = postprocess_dfine_outputs(
        outputs, (orig_w, orig_h), conf_threshold, device
    )
    
    if len(boxes) == 0:
        print(f"\n❌ No objects detected with confidence > {conf_threshold}")
        print(f"   Try lowering --conf-threshold")
        return False
    
    # Print detections
    print(f"\n📊 Final Detections:")
    class_counts = {}
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        class_idx = int(label)
        class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        if i < 10:  # Print first 10
            print(f"   {i+1}. {class_name}: {score:.3f}")
    
    if len(boxes) > 10:
        print(f"   ... and {len(boxes) - 10} more")
    
    print(f"\n   Summary:")
    for class_name, count in sorted(class_counts.items()):
        print(f"   - {class_name}: {count}")
    
    # Visualize
    print(f"\n🎨 Creating visualization...")
    
    # Scale boxes to 640x640 for visualization
    scale_x = 640 / orig_w
    scale_y = 640 / orig_h
    viz_boxes = boxes.cpu().numpy().copy()
    viz_boxes[:, [0, 2]] *= scale_x
    viz_boxes[:, [1, 3]] *= scale_y
    
    visualize_detections(
        img_resized,
        viz_boxes,
        scores.cpu().numpy(),
        labels.cpu().numpy(),
        output_path,
        model_name
    )
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test DFINE Detection')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--config', default='models/dfine_hgnetv2_x_obj2coco.yml',
                        help='DFINE config file')
    parser.add_argument('--baseline-checkpoint', default='models/dfine_0.73.pth',
                        help='Baseline DFINE checkpoint (segmentation model)')
    parser.add_argument('--amodal-checkpoint', default='outputs/amodal_humans/best_model.pth',
                        help='Amodal-trained checkpoint')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--output-dir', default='test_outputs',
                        help='Output directory for visualizations')
    parser.add_argument('--test-baseline', action='store_true',
                        help='Test baseline model')
    parser.add_argument('--test-amodal', action='store_true',
                        help='Test amodal-trained model')
    args = parser.parse_args()
    
    # Default: test both if neither specified
    if not args.test_baseline and not args.test_amodal:
        args.test_baseline = True
        args.test_amodal = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print("🔍 DFINE DETECTION TEST")
    print(f"{'='*80}")
    print(f"Image: {args.image}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"{'='*80}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = []
    
    # Test baseline model
    if args.test_baseline:
        if os.path.exists(args.baseline_checkpoint):
            try:
                model = load_dfine_model(args.config, args.baseline_checkpoint, device)
                output_path = os.path.join(args.output_dir, 'baseline_dfine_result.png')
                success = test_model(
                    model, args.image, args.conf_threshold, device,
                    output_path, 'Baseline DFINE Segmentation (dfine_0.73.pth)'
                )
                results.append(('Baseline', success))
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"\n❌ Error testing baseline model: {e}")
                import traceback
                traceback.print_exc()
                results.append(('Baseline', False))
        else:
            print(f"\n⚠️  Baseline checkpoint not found: {args.baseline_checkpoint}")
            results.append(('Baseline', False))
    
    # Test amodal-trained model
    if args.test_amodal:
        if os.path.exists(args.amodal_checkpoint):
            try:
                model = load_dfine_model(args.config, args.amodal_checkpoint, device)
                output_path = os.path.join(args.output_dir, 'amodal_trained_dfine_result.png')
                success = test_model(
                    model, args.image, args.conf_threshold, device,
                    output_path, 'Amodal-Trained DFINE (outputs/amodal_humans/best_model.pth)'
                )
                results.append(('Amodal-trained', success))
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"\n❌ Error testing amodal-trained model: {e}")
                import traceback
                traceback.print_exc()
                results.append(('Amodal-trained', False))
        else:
            print(f"\n⚠️  Amodal checkpoint not found: {args.amodal_checkpoint}")
            results.append(('Amodal-trained', False))
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    for model_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {model_name}")
    print(f"{'='*80}")
    print(f"\n💡 Check results in: {args.output_dir}/")
    print()


if __name__ == '__main__':
    main()