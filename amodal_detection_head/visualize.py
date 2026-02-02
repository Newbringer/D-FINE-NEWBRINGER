#!/usr/bin/env python3
"""
Visualization Script for Amodal Detection Predictions
Shows visible boxes (green), amodal boxes (red), and occlusion scores
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import cv2

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'segmentation_sivert'))
sys.path.insert(0, str(PROJECT_ROOT / 'glass_wall_detection' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'amodal_detection_head'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

from ochuman_dataset import OCHumanAmodalDataset
from amodal_head import AmodalDetectionHead, CombinedDFINEAmodalModel
from segmentation_sivert.core.models import load_pretrained_dfine, get_actual_backbone_channels
from model_architecture import SegmentationHead


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Amodal Detection Predictions')
    
    parser.add_argument('--ochuman-root', required=True,
                        help='Path to OCHuman dataset root')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to trained amodal model checkpoint')
    parser.add_argument('--dfine-config', default='models/dfine_hgnetv2_x_obj2coco.yml',
                        help='Path to DFINE config')
    parser.add_argument('--dfine-checkpoint', default='models/dfine_0.73.pth',
                        help='Path to DFINE checkpoint')
    parser.add_argument('--seg-checkpoint', required=True,
                        help='Path to segmentation checkpoint')
    parser.add_argument('--split', default='val', choices=['train', 'val'],
                        help='Dataset split to visualize')
    parser.add_argument('--num-images', type=int, default=10,
                        help='Number of images to visualize')
    parser.add_argument('--output-dir', default='visualizations/amodal',
                        help='Directory to save visualizations')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                        help='Confidence threshold for showing predictions')
    parser.add_argument('--device', default='cuda',
                        help='Device to use')
    
    return parser.parse_args()


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


def visualize_sample(image, gt_visible, gt_amodal, pred_visible, pred_amodal, 
                     pred_conf, pred_occ, gt_occ, valid_mask, conf_threshold=0.3):
    """
    Create visualization comparing ground truth and predictions
    
    Args:
        image: [3, H, W] tensor
        gt_visible: [N, 4] ground truth visible boxes
        gt_amodal: [N, 4] ground truth amodal boxes
        pred_visible: [N, 4] predicted visible boxes
        pred_amodal: [N, 4] predicted amodal boxes
        pred_conf: [N] prediction confidence scores
        pred_occ: [N] predicted occlusion scores
        gt_occ: [N] ground truth occlusion scores
        valid_mask: [N] binary mask for valid objects
        conf_threshold: only show predictions above this confidence
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Denormalize image
    img_np = denormalize_image(image)
    img_size = image.shape[-1]
    
    # Ground Truth
    ax = axes[0]
    ax.imshow(img_np)
    ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Get valid ground truth boxes
    valid_indices = torch.where(valid_mask > 0)[0]
    
    if len(valid_indices) > 0:
        gt_vis_boxes = box_cxcywh_to_xyxy(gt_visible[valid_indices], img_size)
        gt_amod_boxes = box_cxcywh_to_xyxy(gt_amodal[valid_indices], img_size)
        gt_occ_scores = gt_occ[valid_indices]
        
        for i, idx in enumerate(valid_indices):
            # Amodal box (red, dashed)
            x1, y1, x2, y2 = gt_amod_boxes[i]
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2, edgecolor='red', facecolor='none',
                linestyle='--', label='Amodal' if i == 0 else ''
            )
            ax.add_patch(rect)
            
            # Visible box (green, solid)
            x1, y1, x2, y2 = gt_vis_boxes[i]
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2, edgecolor='lime', facecolor='none',
                label='Visible' if i == 0 else ''
            )
            ax.add_patch(rect)
            
            # Show occlusion score
            occ_score = gt_occ_scores[i].item()
            ax.text(x1, y1 - 5, f'Occ: {occ_score:.2f}',
                   color='white', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.legend(loc='upper right')
    
    # Predictions
    ax = axes[1]
    ax.imshow(img_np)
    ax.set_title('Predictions', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Filter predictions by confidence
    conf_mask = pred_conf.squeeze() > conf_threshold
    if conf_mask.any():
        pred_vis_boxes = box_cxcywh_to_xyxy(pred_visible[conf_mask], img_size)
        pred_amod_boxes = box_cxcywh_to_xyxy(pred_amodal[conf_mask], img_size)
        pred_conf_scores = pred_conf[conf_mask].squeeze()
        pred_occ_scores = pred_occ[conf_mask].squeeze()
        
        for i in range(len(pred_vis_boxes)):
            # Amodal box (red, dashed)
            x1, y1, x2, y2 = pred_amod_boxes[i]
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2, edgecolor='red', facecolor='none',
                linestyle='--', label='Amodal' if i == 0 else ''
            )
            ax.add_patch(rect)
            
            # Visible box (green, solid)
            x1, y1, x2, y2 = pred_vis_boxes[i]
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2, edgecolor='lime', facecolor='none',
                label='Visible' if i == 0 else ''
            )
            ax.add_patch(rect)
            
            # Show confidence and occlusion
            conf = pred_conf_scores[i].item() if pred_conf_scores.dim() > 0 else pred_conf_scores.item()
            occ = pred_occ_scores[i].item() if pred_occ_scores.dim() > 0 else pred_occ_scores.item()
            
            ax.text(x1, y1 - 5, f'Conf: {conf:.2f} | Occ: {occ:.2f}',
                   color='white', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    else:
        ax.text(0.5, 0.5, 'No predictions above threshold',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=16, color='red', fontweight='bold')
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def load_model(args):
    """Load the trained amodal detection model"""
    print("🚀 Loading model...")
    
    # Load DFINE
    dfine_model = load_pretrained_dfine(args.dfine_config, args.dfine_checkpoint)
    backbone_channels = get_actual_backbone_channels(dfine_model)
    
    # Load segmentation
    seg_checkpoint = torch.load(args.seg_checkpoint, map_location='cpu')
    
    if 'hyperparameters' in seg_checkpoint:
        hyper = seg_checkpoint['hyperparameters']
        feature_dim = hyper.get('feature_dim', 256)
        num_classes = hyper.get('num_classes', 7)
    else:
        feature_dim = 256
        num_classes = 7
    
    segmentation_head = SegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=num_classes,
        feature_dim=feature_dim,
        dropout_rate=0.1
    )
    
    # Load seg weights
    if isinstance(seg_checkpoint, dict) and 'model_state_dict' in seg_checkpoint:
        state_dict = seg_checkpoint['model_state_dict']
    else:
        state_dict = seg_checkpoint
    
    seg_state_dict = {k.replace('seg_head.', ''): v for k, v in state_dict.items() if 'seg_head' in k}
    if seg_state_dict:
        segmentation_head.load_state_dict(seg_state_dict, strict=False)
    
    # Create amodal head
    amodal_head = AmodalDetectionHead(
        in_channels=backbone_channels[-1],
        hidden_dim=256,
        num_classes=2,  # OCHuman: background + person
        num_queries=100
    )
    
    # Load amodal weights
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    amodal_head.load_state_dict(checkpoint['amodal_head_state_dict'])
    
    # Create combined model
    model = CombinedDFINEAmodalModel(
        dfine_model=dfine_model,
        segmentation_head=segmentation_head,
        amodal_head=amodal_head
    )
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"   Best loss: {checkpoint.get('best_loss', 'unknown')}")
    
    return model


def main():
    args = parse_args()
    
    print("=" * 80)
    print("🎨 VISUALIZING AMODAL DETECTION PREDICTIONS")
    print("=" * 80)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"\n📊 Loading OCHuman {args.split} dataset...")
    dataset = OCHumanAmodalDataset(
        root_dir=args.ochuman_root,
        split=args.split,
        image_size=640,
        max_objects=50,
        augment=False
    )
    
    # Load model
    model = load_model(args)
    model = model.to(device)
    model.eval()
    
    print(f"\n🎨 Creating visualizations...")
    print(f"   Confidence threshold: {args.conf_threshold}")
    print(f"   Number of images: {args.num_images}")
    
    # Process images
    with torch.no_grad():
        for i in range(min(args.num_images, len(dataset))):
            print(f"\n📸 Processing image {i+1}/{args.num_images}...")
            
            # Get sample
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            
            # Get ground truth
            gt_visible = sample['visible_boxes']
            gt_amodal = sample['amodal_boxes']
            gt_occ = sample['occlusion_scores']
            valid_mask = sample['valid_mask']
            
            num_valid = valid_mask.sum().item()
            print(f"   Ground truth objects: {num_valid}")
            
            if num_valid > 0:
                valid_occ = gt_occ[valid_mask > 0]
                print(f"   Occlusion range: {valid_occ.min():.2f} - {valid_occ.max():.2f}")
            
            # Get predictions
            outputs = model(image)
            
            pred_visible = outputs['visible_boxes'][0].cpu()
            pred_amodal = outputs['amodal_boxes'][0].cpu()
            pred_conf = outputs['confidence_scores'][0].cpu()
            pred_occ = outputs['occlusion_scores'][0].cpu()
            
            # Count predictions above threshold
            num_preds = (pred_conf.squeeze() > args.conf_threshold).sum().item()
            print(f"   Predictions (conf > {args.conf_threshold}): {num_preds}")
            
            # Create visualization
            fig = visualize_sample(
                sample['image'],
                gt_visible,
                gt_amodal,
                pred_visible,
                pred_amodal,
                pred_conf,
                pred_occ,
                gt_occ,
                valid_mask,
                conf_threshold=args.conf_threshold
            )
            
            # Save
            filename = sample.get('file_name', f'image_{i:04d}.jpg')
            output_path = os.path.join(args.output_dir, f'viz_{i:04d}_{Path(filename).stem}.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"   ✅ Saved: {output_path}")
    
    print(f"\n🎉 Visualization complete!")
    print(f"📁 Saved to: {args.output_dir}")
    
    # Print summary statistics
    print(f"\n📊 Summary:")
    print(f"   Images processed: {min(args.num_images, len(dataset))}")
    print(f"   Output directory: {args.output_dir}")
    print(f"\n💡 Interpretation Guide:")
    print(f"   🟢 Green boxes = Visible parts (what you can see)")
    print(f"   🔴 Red dashed boxes = Amodal (full person including occluded parts)")
    print(f"   Occlusion score: 0 = not occluded, 1 = fully occluded")
    print(f"   Confidence: Model's certainty in the detection")


if __name__ == '__main__':
    main()