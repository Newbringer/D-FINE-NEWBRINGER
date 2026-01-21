#!/usr/bin/env python3
"""
Glass Wall Detection Testing
Test trained model on images
"""

import os
import sys
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path

sys.path.insert(0, '../../src')

from model_utils import load_model_with_segmentation


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


def preprocess_image(image, image_size=640):
    """Preprocess image for inference"""
    orig_h, orig_w = image.shape[:2]
    
    # Resize
    resized = cv2.resize(image, (image_size, image_size))
    
    # Convert to RGB and normalize
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (normalized - mean) / std
    
    # Convert to tensor (NCHW)
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, (orig_w, orig_h)


def postprocess_outputs(outputs, orig_size, conf_threshold=0.5):
    """Postprocess model outputs"""
    # Extract predictions
    if isinstance(outputs, dict):
        boxes = outputs.get('pred_boxes', outputs.get('boxes'))
        scores = outputs.get('pred_scores', outputs.get('scores'))
        labels = outputs.get('pred_labels', outputs.get('labels'))
        
        if boxes is None:
            # Try alternative keys
            for key in outputs.keys():
                if 'box' in key.lower():
                    boxes = outputs[key]
                elif 'score' in key.lower():
                    scores = outputs[key]
                elif 'label' in key.lower() or 'class' in key.lower():
                    labels = outputs[key]
    else:
        # Assume tuple output
        boxes, scores, labels = outputs[:3]
    
    if boxes is None or scores is None or labels is None:
        return np.array([]), np.array([]), np.array([])
    
    # Convert to numpy
    boxes = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
    scores = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    # Handle different shapes
    if len(boxes.shape) == 3:
        boxes = boxes[0]
    if len(scores.shape) == 2:
        scores = scores[0]
    if len(labels.shape) == 2:
        labels = labels[0]
    
    # Filter by confidence
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Scale boxes to original size
    orig_w, orig_h = orig_size
    scale_x = orig_w / 640
    scale_y = orig_h / 640
    
    if len(boxes) > 0:
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
    
    return boxes, scores, labels


def draw_detections(image, boxes, scores, labels, show_all=True):
    """Draw detection boxes on image"""
    vis_image = image.copy()
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        
        # Get class name
        class_idx = int(label)
        class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f'class_{class_idx}'
        
        # Color coding
        if class_idx == 80:  # Glass wall
            color = (0, 255, 0)  # Green
            thickness = 3
        else:
            if not show_all:
                continue
            color = (255, 0, 0)  # Blue for others
            thickness = 2
        
        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label_text = f'{class_name}: {score:.2f}'
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(vis_image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(vis_image, label_text, (x1, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_image


def test_image(model, image_path, conf_threshold=0.5, show_all=True, output_path=None, device='cuda'):
    """Test on single image"""
    print(f"\n📸 Testing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load: {image_path}")
        return
    
    # Preprocess
    input_tensor, orig_size = preprocess_image(image)
    input_tensor = input_tensor.to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Postprocess
    boxes, scores, labels = postprocess_outputs(outputs, orig_size, conf_threshold)
    
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
    
    # Visualize
    vis_image = draw_detections(image, boxes, scores, labels, show_all)
    
    # Save if requested
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"💾 Saved: {output_path}")
    
    # Display
    cv2.imshow('Glass Detection', vis_image)
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_directory(model, input_dir, conf_threshold=0.5, show_all=True, output_dir=None, device='cuda'):
    """Test on directory of images"""
    print(f"\n📂 Testing directory: {input_dir}")
    
    # Get all images
    image_files = list(Path(input_dir).glob('*.jpg')) + \
                 list(Path(input_dir).glob('*.png')) + \
                 list(Path(input_dir).glob('*.jpeg'))
    
    print(f"   Found {len(image_files)} images")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    total_glass = 0
    total_other = 0
    
    for img_path in image_files:
        print(f"\n📸 {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"   ⚠️  Failed to load")
            continue
        
        # Preprocess
        input_tensor, orig_size = preprocess_image(image)
        input_tensor = input_tensor.to(device)
        
        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Postprocess
        boxes, scores, labels = postprocess_outputs(outputs, orig_size, conf_threshold)
        
        # Count
        glass_count = (labels == 80).sum() if len(labels) > 0 else 0
        other_count = (labels != 80).sum() if len(labels) > 0 else 0
        
        total_glass += glass_count
        total_other += other_count
        
        print(f"   Glass: {glass_count}, Other: {other_count}")
        
        # Visualize and save
        if output_dir:
            vis_image = draw_detections(image, boxes, scores, labels, show_all)
            output_path = os.path.join(output_dir, f"result_{img_path.name}")
            cv2.imwrite(output_path, vis_image)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Total glass detections: {total_glass}")
    print(f"   Total other detections: {total_other}")
    print(f"   Avg glass per image: {total_glass/len(image_files):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Test Glass Wall Detection')
    
    # Model
    parser.add_argument('--checkpoint', default='../outputs/glass_detection/best_model.pth',
                       help='Path to trained checkpoint')
    parser.add_argument('--dfine-config', default='../models/dfine_hgnetv2_x_obj2coco.yml',
                       help='DFINE config file')
    
    # Input
    parser.add_argument('--image', default=None,
                       help='Test single image')
    parser.add_argument('--directory', default=None,
                       help='Test all images in directory')
    
    # Options
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--show-all', action='store_true',
                       help='Show all detections (not just glass)')
    parser.add_argument('--output', default=None,
                       help='Output file or directory')
    parser.add_argument('--device', default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load model
    print("📦 Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = load_model_with_segmentation(
        args.checkpoint,
        args.dfine_config,
        device
    )
    
    # Test
    if args.image:
        test_image(
            model, args.image,
            conf_threshold=args.confidence,
            show_all=args.show_all,
            output_path=args.output,
            device=device
        )
    elif args.directory:
        test_directory(
            model, args.directory,
            conf_threshold=args.confidence,
            show_all=args.show_all,
            output_dir=args.output,
            device=device
        )
    else:
        print("❌ Error: Must provide --image or --directory")


if __name__ == '__main__':
    main()