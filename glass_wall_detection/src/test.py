#!/usr/bin/env python3
"""
Glass Wall Detection Testing
Test trained model on images with visualization
"""

import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# Add DFINE to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    sys.path.insert(0, project_root)

# Import src module first to trigger all registrations
import src
from src.core import YAMLConfig

# Import dataset for validation testing
from dataset import GlassWallDataset


# COCO + Glass class names
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
    'toothbrush', 'glass_wall'  # Class 80
]


def expand_model_to_81_classes(model):
    """Expand model from 80 to 81 classes - same as training"""
    print("🔧 Expanding detection head: 80 -> 81 classes")
    
    for name, module in model.named_modules():
        # Expand classification layers
        if any(key in name.lower() for key in ['class_embed', 'cls', 'score']):
            if isinstance(module, nn.Linear) and module.out_features == 80:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                new_module = nn.Linear(module.in_features, 81)
                with torch.no_grad():
                    new_module.weight[:80] = module.weight
                    if module.bias is not None:
                        new_module.bias[:80] = module.bias
                    nn.init.normal_(new_module.weight[80:], mean=0, std=0.01)
                    if new_module.bias is not None:
                        nn.init.zeros_(new_module.bias[80:])
                
                setattr(parent, child_name, new_module)
                print(f"   ✅ {name}: 80 -> 81 classes")
            
            # Expand denoising embedding
            elif isinstance(module, nn.Embedding) and module.num_embeddings == 81:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                padding_idx = module.padding_idx if hasattr(module, 'padding_idx') else None
                new_module = nn.Embedding(82, module.embedding_dim, padding_idx=81)
                
                with torch.no_grad():
                    new_module.weight[:81] = module.weight
                    if padding_idx != 81:
                        nn.init.normal_(new_module.weight[81:82], mean=0, std=0.01)
                
                setattr(parent, child_name, new_module)
                print(f"   ✅ {name}: 81 -> 82 embeddings")
    
    # Update num_classes attributes
    if hasattr(model, 'num_classes'):
        model.num_classes = 81
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'num_classes'):
        model.decoder.num_classes = 81
    
    return model


def load_trained_model(checkpoint_path, config_path, device):
    """Load trained model with proper expansion based on checkpoint."""
    print(f"📦 Loading model from {checkpoint_path}")
    
    # Load checkpoint first to infer class count
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Infer number of classes from checkpoint (80 or 81)
    target_classes = 80
    for key, value in state_dict.items():
        if 'dec_score_head.0.weight' in key and hasattr(value, 'shape'):
            if value.shape[0] == 81:
                target_classes = 81
            break

    # Load DFINE model
    cfg = YAMLConfig(config_path)
    model = cfg.model
    postprocessor = cfg.postprocessor

    # Expand only when checkpoint expects 81 classes
    if target_classes == 81:
        model = expand_model_to_81_classes(model)
        if hasattr(postprocessor, 'num_classes'):
            postprocessor.num_classes = 81
        print("✅ Detected 81-class checkpoint")
    else:
        if hasattr(postprocessor, 'num_classes'):
            postprocessor.num_classes = 80
        print("✅ Detected 80-class checkpoint")
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    return model, postprocessor


def preprocess_image(image_path, image_size=640):
    """Load and preprocess image"""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    orig_h, orig_w = image.shape[:2]
    
    # Resize
    resized = cv2.resize(image, (image_size, image_size))
    
    # Convert BGR to RGB and normalize
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    
    # Convert to tensor (NCHW) - ensure float32
    tensor = torch.from_numpy(normalized.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, image, (orig_w, orig_h)


def postprocess_predictions(outputs, postprocessor, orig_size, conf_threshold=0.3, device='cuda'):
    """Postprocess DFINE outputs to get boxes, scores, labels.
    
    Use DFINE-style top-k selection across all queries/classes to reduce noise.
    """
    orig_w, orig_h = orig_size
    
    # Extract raw predictions from DFINE output
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4] in cxcywh format
    
    num_classes = pred_logits.shape[-1]
    num_top_queries = getattr(postprocessor, 'num_top_queries', 300)
    
    # Convert boxes from cxcywh (normalized) to xyxy (pixel coordinates)
    cx, cy, w, h = pred_boxes.unbind(dim=-1)
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    
    # DFINE-style top-k across all classes/queries
    scores = pred_logits.sigmoid()
    scores, index = torch.topk(scores.flatten(), k=min(num_top_queries, scores.numel()))
    labels = torch.remainder(index, num_classes)
    index = index // num_classes
    boxes = boxes_xyxy[index]
    
    # Filter by confidence
    keep = scores > conf_threshold
    boxes = boxes[keep].cpu().numpy() if keep.any() else np.array([])
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()
    
    return boxes, scores, labels


def draw_detections(image, boxes, scores, labels, show_all=True, highlight_glass=True):
    """Draw detection boxes on image"""
    vis_image = image.copy()
    
    # Draw glass detections last (on top)
    glass_detections = []
    other_detections = []
    
    for box, score, label in zip(boxes, scores, labels):
        if int(label) == 80:
            glass_detections.append((box, score, label))
        else:
            other_detections.append((box, score, label))
    
    # Draw others first
    if show_all:
        for box, score, label in other_detections:
            x1, y1, x2, y2 = box.astype(int)
            class_idx = int(label)
            class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
            
            color = (255, 0, 0)  # Blue for others
            thickness = 2
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            label_text = f'{class_name}: {score:.2f}'
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(vis_image, label_text, (x1, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw glass detections on top (highlighted)
    for box, score, label in glass_detections:
        x1, y1, x2, y2 = box.astype(int)
        
        color = (0, 255, 0)  # Green for glass
        thickness = 3 if highlight_glass else 2
        
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
        
        label_text = f'GLASS: {score:.2f}'
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_image, (x1, y1 - text_h - 6), (x1 + text_w, y1), color, -1)
        cv2.putText(vis_image, label_text, (x1, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_image


def test_single_image(model, postprocessor, image_path, conf_threshold, show_all, output_path, device):
    """Test on a single image"""
    print(f"\n📸 Testing: {image_path}")
    
    # Preprocess
    input_tensor, orig_image, orig_size = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Postprocess
    boxes, scores, labels = postprocess_predictions(outputs, postprocessor, orig_size, conf_threshold, device)
    
    # Count detections
    glass_count = (labels == 80).sum() if len(labels) > 0 else 0
    other_count = (labels != 80).sum() if len(labels) > 0 else 0
    
    print(f"📊 Results:")
    print(f"   Glass detections: {glass_count}")
    print(f"   Other detections: {other_count}")
    
    # Print detections
    if len(boxes) > 0:
        print(f"\n🎯 Detections:")
        for box, score, label in zip(boxes, scores, labels):
            class_idx = int(label)
            class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
            print(f"   {class_name}: {score:.3f} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
    else:
        print("   No detections above threshold")
    
    # Visualize
    vis_image = draw_detections(orig_image, boxes, scores, labels, show_all)
    
    # Save (always save, no GUI display for server environments)
    if not output_path:
        # Auto-generate output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_detection.jpg"
    
    cv2.imwrite(output_path, vis_image)
    print(f"💾 Saved: {output_path}")
    
    return glass_count, other_count


def test_validation_set(model, postprocessor, data_path, conf_threshold, show_all, output_dir, device):
    """Test on validation dataset"""
    print(f"\n📂 Testing validation set: {data_path}")
    
    # Load validation dataset
    val_dataset = GlassWallDataset(data_path, split='val', class_id=80)
    print(f"   Found {len(val_dataset)} validation images")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    total_glass = 0
    total_other = 0
    total_gt_glass = 0
    
    for idx in tqdm(range(len(val_dataset)), desc="Testing"):
        # Get image info and path from dataset
        img_info = val_dataset.images[idx]
        img_path = val_dataset.root_dir / img_info['file_name']
        
        try:
            # Get data from dataset (already preprocessed)
            image_tensor, targets = val_dataset[idx]
            
            # Ground truth boxes (in normalized cxcywh format from dataset)
            gt_boxes = targets['boxes'].cpu().numpy() if len(targets['boxes']) > 0 else []
            total_gt_glass += len(gt_boxes)
            
            # Also load original image for visualization
            orig_image = cv2.imread(str(img_path))
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            orig_size = (orig_image.shape[1], orig_image.shape[0])  # (width, height)
            
            # Prepare input
            input_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
            
            # Postprocess
            boxes, scores, labels = postprocess_predictions(outputs, postprocessor, orig_size, conf_threshold, device)
            
            # Count
            glass_count = (labels == 80).sum() if len(labels) > 0 else 0
            other_count = (labels != 80).sum() if len(labels) > 0 else 0
            
            total_glass += glass_count
            total_other += other_count
            
            # Visualize and save
            if output_dir:
                vis_image = draw_detections(orig_image, boxes, scores, labels, show_all)
                
                # Also draw ground truth in yellow dashed
                # Convert gt_boxes from normalized cxcywh to pixel xyxy
                for gt_box in gt_boxes:
                    cx, cy, w, h = gt_box
                    x1 = int((cx - w/2) * orig_size[0])
                    y1 = int((cy - h/2) * orig_size[1])
                    x2 = int((cx + w/2) * orig_size[0])
                    y2 = int((cy + h/2) * orig_size[1])
                    
                    # Yellow dashed rectangle for GT
                    for i in range(0, int(x2-x1), 10):
                        cv2.line(vis_image, (x1+i, y1), (min(x1+i+5, x2), y1), (0, 255, 255), 2)
                        cv2.line(vis_image, (x1+i, y2), (min(x1+i+5, x2), y2), (0, 255, 255), 2)
                    for i in range(0, int(y2-y1), 10):
                        cv2.line(vis_image, (x1, y1+i), (x1, min(y1+i+5, y2)), (0, 255, 255), 2)
                        cv2.line(vis_image, (x2, y1+i), (x2, min(y1+i+5, y2)), (0, 255, 255), 2)
                
                # Convert RGB to BGR for saving
                vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                output_path = os.path.join(output_dir, f"result_{Path(img_path).name}")
                cv2.imwrite(output_path, vis_image_bgr)
        
        except Exception as e:
            print(f"   ⚠️  Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n📊 Validation Summary:")
    print(f"   Images tested: {len(val_dataset)}")
    print(f"   Ground truth glass: {total_gt_glass}")
    print(f"   Predicted glass: {total_glass}")
    print(f"   Other detections: {total_other}")
    print(f"   Avg glass per image: {total_glass/len(val_dataset):.1f}")
    print(f"   Recall estimate: {total_glass/max(total_gt_glass, 1):.1%}")


def main():
    parser = argparse.ArgumentParser(description='Test Glass Wall Detection Model')
    
    # Model
    parser.add_argument('--checkpoint', default='../outputs/glass_detection_proper/dfine_0.73.pth',
                       help='Path to trained checkpoint')
    parser.add_argument('--config', default='../models/dfine_hgnetv2_x_obj2coco.yml',
                       help='DFINE config file')
    
    # Input (choose one)
    parser.add_argument('--image', default=None,
                       help='Test single image')
    parser.add_argument('--validation', default=None,
                       help='Test on validation set (path to glass_wall data)')
    
    # Options
    parser.add_argument('--confidence', type=float, default=0.2,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--show-all', action='store_true',
                       help='Show all COCO detections (not just glass)')
    parser.add_argument('--output', default=None,
                       help='Output file (for --image) or directory (for --validation)')
    parser.add_argument('--device', default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GLASS WALL DETECTION - TESTING")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, postprocessor = load_trained_model(args.checkpoint, args.config, device)
    
    # Test
    if args.image:
        test_single_image(
            model, postprocessor, args.image,
            args.confidence, args.show_all, args.output, device
        )
    elif args.validation:
        output_dir = args.output or '../outputs/validation_results'
        test_validation_set(
            model, postprocessor, args.validation,
            args.confidence, args.show_all, output_dir, device
        )
    else:
        print("❌ Error: Must provide --image or --validation")
        print("\nExamples:")
        print("  Test single image:")
        print("    python test.py --image path/to/image.jpg --output result.jpg")
        print("\n  Test validation set:")
        print("    python test.py --validation ../data/glass_wall --output ../outputs/val_results")
        print("\n  Show all COCO detections:")
        print("    python test.py --image image.jpg --show-all")


if __name__ == '__main__':
    main()
