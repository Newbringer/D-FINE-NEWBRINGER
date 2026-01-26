#!/usr/bin/env python3
"""
Live webcam inference using standard DFINE object detection model.
No custom modifications - just pure DFINE inference.
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

# Add DFINE to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    sys.path.insert(0, project_root)

# Import src module first to trigger all registrations
import src  # noqa: F401
from src.core import YAMLConfig


# COCO class names (80 classes)
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


def load_model(config_path, checkpoint_path, device):
    """Load DFINE model, detecting whether it's 80 or 81 classes."""
    print(f"📦 Loading DFINE model...")
    print(f"   Config: {config_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # Load checkpoint first to detect number of classes
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict (prefer EMA if available)
    if 'ema' in checkpoint and 'module' in checkpoint['ema']:
        state_dict = checkpoint['ema']['module']
        print("   Using EMA weights")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Check if this is a segmentation checkpoint (has dfine_model.* or seg_head.* prefix)
    is_segmentation_checkpoint = any('dfine_model.' in key or 'seg_head.' in key for key in state_dict.keys())
    
    if is_segmentation_checkpoint:
        print("   Detected segmentation checkpoint - extracting DFINE weights...")
        # Extract only dfine_model.* weights and remove prefix
        dfine_weights = {}
        for key, value in state_dict.items():
            if key.startswith('dfine_model.'):
                # Remove 'dfine_model.' prefix
                new_key = key[len('dfine_model.'):]
                dfine_weights[new_key] = value
        
        if len(dfine_weights) == 0:
            print("   ⚠️  No dfine_model.* weights found, trying without prefix...")
            dfine_weights = {k: v for k, v in state_dict.items() if not k.startswith('seg_head.')}
        
        state_dict = dfine_weights
        print(f"   Extracted {len(state_dict)} DFINE weights (removed segmentation layers)")
    
    # Detect number of classes from checkpoint
    num_classes = 80  # default
    for key, value in state_dict.items():
        # Look for classification head output features
        # Skip denoising_class_embed as it's num_classes + 1
        if 'denoising_class_embed' in key:
            continue
        
        # Look for actual classification heads (dec_score_head or enc_score_head)
        if ('dec_score_head' in key or 'enc_score_head' in key) and 'weight' in key:
            if hasattr(value, 'shape') and len(value.shape) > 0:
                detected_classes = value.shape[0]
                if detected_classes in [80, 81]:
                    num_classes = detected_classes
                    print(f"   Detected {num_classes} classes from {key}")
                    break
    
    if num_classes == 80:
        print(f"   Using default 80 classes (standard COCO)")
    
    # Load config
    cfg = YAMLConfig(config_path)
    model = cfg.model
    postprocessor = cfg.postprocessor
    
    # Expand model to 81 classes if checkpoint expects it
    if num_classes == 81:
        print("   Expanding model to 81 classes...")
        model = expand_model_to_81_classes(model)
        # Update postprocessor
        if hasattr(postprocessor, 'num_classes'):
            postprocessor.num_classes = 81
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   ⚠️  Missing keys: {len(missing)}")
        if len(missing) <= 10:
            for key in missing:
                print(f"      - {key}")
    if unexpected:
        print(f"   ⚠️  Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 10:
            for key in unexpected:
                print(f"      - {key}")
    
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Number of classes: {num_classes}")
    
    return model, postprocessor, num_classes


def expand_model_to_81_classes(model):
    """Expand detection head from 80 to 81 classes."""
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
                print(f"      ✅ {name}: 80 -> 81 classes")
            
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
                print(f"      ✅ {name}: 81 -> 82 embeddings")
    
    # Update num_classes attributes
    if hasattr(model, 'num_classes'):
        model.num_classes = 81
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'num_classes'):
        model.decoder.num_classes = 81
    
    return model


def preprocess_frame(frame_bgr, image_size=640):
    """Preprocess frame for DFINE inference."""
    orig_h, orig_w = frame_bgr.shape[:2]
    
    # Resize to square
    resized = cv2.resize(frame_bgr, (image_size, image_size))
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    
    # Convert to tensor (NCHW)
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, (orig_w, orig_h)


def postprocess_outputs(outputs, postprocessor, orig_size, conf_threshold, device, verbose=False):
    """Postprocess DFINE outputs with manual top-k selection and NMS."""
    orig_w, orig_h = orig_size
    
    if verbose:
        print("\n" + "="*60)
        print("POSTPROCESSING DEBUG")
        print("="*60)
        print(f"Original size: {orig_w}x{orig_h}")
        print(f"Confidence threshold: {conf_threshold}")
    
    # Extract raw predictions from DFINE output
    if verbose:
        print(f"\nOutput keys: {outputs.keys()}")
    
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4] in cxcywh normalized format
    
    if verbose:
        print(f"pred_logits shape: {pred_logits.shape}")
        print(f"pred_boxes shape: {pred_boxes.shape}")
        print(f"pred_logits dtype: {pred_logits.dtype}")
        print(f"pred_boxes dtype: {pred_boxes.dtype}")
        print(f"pred_logits range: [{pred_logits.min():.3f}, {pred_logits.max():.3f}]")
        print(f"pred_boxes range: [{pred_boxes.min():.3f}, {pred_boxes.max():.3f}]")
    
    num_classes = pred_logits.shape[-1]
    
    if verbose:
        print(f"\nNumber of classes: {num_classes}")
        print(f"Number of queries: {pred_logits.shape[0]}")
    
    # Get num_top_queries from postprocessor (typically 300)
    num_top_queries = getattr(postprocessor, 'num_top_queries', 300)
    
    if verbose:
        print(f"num_top_queries: {num_top_queries}")
    
    # Convert boxes from cxcywh (normalized 0-1) to xyxy (pixel coordinates)
    cx, cy, w, h = pred_boxes.unbind(dim=-1)
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    
    if verbose:
        print(f"\nBoxes after converting to pixel coordinates:")
        print(f"  boxes_xyxy shape: {boxes_xyxy.shape}")
        print(f"  boxes_xyxy range: [{boxes_xyxy.min():.1f}, {boxes_xyxy.max():.1f}]")
    
    # Apply sigmoid to logits to get scores
    scores = pred_logits.sigmoid()
    
    if verbose:
        print(f"\nScores after sigmoid:")
        print(f"  scores shape: {scores.shape}")
        print(f"  scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  scores mean: {scores.mean():.4f}")
        print(f"  scores > 0.1: {(scores > 0.1).sum().item()}")
        print(f"  scores > {conf_threshold}: {(scores > conf_threshold).sum().item()}")
        
        # Show top 10 raw scores
        top_scores_flat, top_indices_flat = scores.flatten().topk(10)
        print(f"\n  Top 10 raw scores (before filtering):")
        for i, (score, idx) in enumerate(zip(top_scores_flat, top_indices_flat)):
            query_idx = idx.item() // num_classes
            class_idx = idx.item() % num_classes
            class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
            print(f"    {i+1}. Query {query_idx}, Class {class_idx} ({class_name}): {score.item():.4f}")
    
    # DFINE-style top-k selection across all queries and classes
    # This flattens [num_queries, num_classes] and takes top-k
    scores_flat = scores.flatten()
    topk_values, topk_indices = torch.topk(
        scores_flat, 
        k=min(num_top_queries, scores_flat.numel())
    )
    
    if verbose:
        print(f"\nAfter top-k selection (k={min(num_top_queries, scores_flat.numel())}):")
        print(f"  topk_values shape: {topk_values.shape}")
        print(f"  topk_values range: [{topk_values.min():.4f}, {topk_values.max():.4f}]")
        print(f"  topk_values > {conf_threshold}: {(topk_values > conf_threshold).sum().item()}")
    
    # Convert flat indices back to (query_idx, class_idx)
    labels = torch.remainder(topk_indices, num_classes)
    query_indices = topk_indices // num_classes
    boxes = boxes_xyxy[query_indices]
    scores = topk_values
    
    if verbose:
        print(f"\nAfter converting indices:")
        print(f"  boxes shape: {boxes.shape}")
        print(f"  scores shape: {scores.shape}")
        print(f"  labels shape: {labels.shape}")
    
    # Filter by confidence threshold
    keep = scores > conf_threshold
    
    if verbose:
        print(f"\nAfter confidence filtering (threshold={conf_threshold}):")
        print(f"  keep count: {keep.sum().item()} / {len(keep)}")
    
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    if verbose:
        print(f"  boxes shape after filter: {boxes.shape}")
        print(f"  scores shape after filter: {scores.shape}")
        print(f"  labels shape after filter: {labels.shape}")
        
        if len(scores) > 0:
            print(f"\n  Detections before NMS:")
            for i in range(min(10, len(scores))):
                class_idx = int(labels[i])
                class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
                print(f"    {i+1}. {class_name}: {scores[i].item():.4f} at [{boxes[i][0]:.0f},{boxes[i][1]:.0f},{boxes[i][2]:.0f},{boxes[i][3]:.0f}]")
    
    # Apply NMS per class to remove duplicates
    if len(boxes) > 0:
        keep_indices = []
        for class_id in torch.unique(labels):
            class_mask = labels == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = torch.where(class_mask)[0]
            
            if verbose:
                print(f"\n  NMS for class {class_id}: {len(class_boxes)} boxes")
            
            # NMS with IoU threshold 0.65 (standard for COCO)
            from torchvision.ops import nms
            nms_keep = nms(
                class_boxes.float(),
                class_scores.float(),
                0.65
            )
            
            if verbose:
                print(f"    After NMS: {len(nms_keep)} boxes kept")
            
            keep_indices.append(class_indices[nms_keep])
        
        if keep_indices:
            keep_indices = torch.cat(keep_indices)
            boxes = boxes[keep_indices].cpu().numpy()
            scores = scores[keep_indices].cpu().numpy()
            labels = labels[keep_indices].cpu().numpy()
            
            if verbose:
                print(f"\n  Final detections after NMS: {len(boxes)}")
        else:
            boxes = np.array([])
            scores = np.array([])
            labels = np.array([])
            
            if verbose:
                print(f"\n  No detections kept after NMS")
    else:
        boxes = np.array([])
        scores = np.array([])
        labels = np.array([])
        
        if verbose:
            print(f"\n  No detections after confidence filtering")
    
    if verbose:
        print("="*60)
        print()
    
    return boxes, scores, labels


def draw_detections(frame, boxes, scores, labels, show_labels=True):
    """Draw bounding boxes on frame."""
    vis_frame = frame.copy()
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        
        # Color based on class
        color = (0, 255, 0)  # Green
        
        # Draw box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if show_labels:
            class_idx = int(label)
            class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
            label_text = f'{class_name}: {score:.2f}'
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            
            # Text
            cv2.putText(vis_frame, label_text, (x1, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_frame


def parse_args():
    parser = argparse.ArgumentParser(description='DFINE Live Webcam Object Detection')
    parser.add_argument('--checkpoint', default='../outputs/glass_detection_segmentation/best_glass_model.pth',
                        help='Path to DFINE checkpoint')
    parser.add_argument('--config', default='../models/dfine_hgnetv2_x_obj2coco.yml',
                        help='DFINE config file')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Webcam device ID')
    parser.add_argument('--width', type=int, default=1280,
                        help='Capture width')
    parser.add_argument('--height', type=int, default=720,
                        help='Capture height')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    parser.add_argument('--flip', action='store_true',
                        help='Flip frame horizontally')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detection information')
    parser.add_argument('--debug', action='store_true',
                        help='Enable detailed debugging output for postprocessing')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable GUI display')
    parser.add_argument('--output', default=None,
                        help='Output video file path')
    parser.add_argument('--only-class-81', action='store_true',
                        help='Only keep detections for class 81 (label index 80)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("DFINE OBJECT DETECTION - LIVE WEBCAM")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.confidence}")
    print("=" * 80)
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, postprocessor, num_classes = load_model(args.config, args.checkpoint, device)
    
    # Update COCO_CLASSES if we have 81 classes
    global COCO_CLASSES
    if num_classes == 81 and len(COCO_CLASSES) == 80:
        COCO_CLASSES = COCO_CLASSES + ['glass_wall']
        print(f"   Added 'glass_wall' as class 80")
    
    print(f"   Using {len(COCO_CLASSES)} class names")
    
    # Open webcam
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera_id}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Setup display
    window_name = "DFINE Object Detection (Press Q to quit)"
    display_enabled = not args.no_display
    if display_enabled:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error:
            print("⚠️  OpenCV GUI not available. Running in headless mode.")
            display_enabled = False
    
    # Setup video writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, 20.0, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {args.output}")
    
    # FPS calculation
    last_time = time.time()
    fps = 0.0
    
    print("\n🎥 Starting webcam inference...")
    print("Press 'q' to quit\n")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️  Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Flip if requested
            if args.flip:
                frame = cv2.flip(frame, 1)
            
            # Preprocess
            input_tensor, orig_size = preprocess_frame(frame)
            input_tensor = input_tensor.to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
            
            # Postprocess
            boxes, scores, labels = postprocess_outputs(
                outputs, postprocessor, orig_size, args.confidence, device, 
                verbose=args.debug
            )

            if args.only_class_81:
                class_id = 80
                if len(labels) > 0:
                    class_mask = labels == class_id
                    boxes = boxes[class_mask]
                    scores = scores[class_mask]
                    labels = labels[class_mask]
            
            # Verbose output
            if args.verbose and len(boxes) > 0:
                print(f"\n Frame detections ({len(boxes)} total):")
                for box, score, label in zip(boxes[:5], scores[:5], labels[:5]):  # Show top 5
                    class_idx = int(label)
                    class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
                    print(f"  {class_name}: {score:.3f}")
            
            # Draw detections
            vis_frame = draw_detections(frame, boxes, scores, labels)
            
            # Calculate and draw FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
            last_time = now
            
            cv2.putText(
                vis_frame,
                f"FPS: {fps:.1f} | Detections: {len(boxes)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            # Save to video
            if writer:
                writer.write(vis_frame)
            
            # Display
            if display_enabled:
                cv2.imshow(window_name, vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display_enabled:
            cv2.destroyAllWindows()
        
        print("\n✅ Webcam inference stopped")


if __name__ == '__main__':
    main()