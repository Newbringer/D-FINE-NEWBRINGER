#!/usr/bin/env python3
"""
Train Amodal Detection Head for Occluded Humans
Uses frozen DFINE backbone + trainable amodal head
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'amodal_detection_head'))
sys.path.insert(0, str(PROJECT_ROOT))

from cocoa_dataset import COCOAAmodalDataset, collate_fn
from amodal_head import AmodalOffsetHead, AmodalLoss, extract_roi_features


def load_dfine_model(config_path, checkpoint_path, device):
    """Load pretrained DFINE model"""
    try:
        # Try loading from segmentation_sivert
        sys.path.insert(0, str(PROJECT_ROOT / 'segmentation_sivert'))
        from core.models import load_pretrained_dfine
        model = load_pretrained_dfine(config_path, checkpoint_path)
    except:
        # Fallback to manual loading
        try:
            from src.core import YAMLConfig
        except:
            from core import YAMLConfig
        
        cfg = YAMLConfig(str(config_path))
        model = cfg.model
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'ema' in checkpoint and 'module' in checkpoint['ema']:
            state_dict = checkpoint['ema']['module']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
    
    return model.to(device)


def get_backbone_channels(model, device):
    """Get backbone output channels"""
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 640, 640, device=device)
        features = model.backbone(dummy)
        if isinstance(features, (list, tuple)):
            channels = [f.shape[1] for f in features]
        else:
            channels = [features.shape[1]]
    return channels


def parse_args():
    parser = argparse.ArgumentParser(description='Train Amodal Detection Head')
    
    # Data
    parser.add_argument('--train-images', default='coco/train2014',
                        help='Path to training images')
    parser.add_argument('--val-images', default='coco/val2014',
                        help='Path to validation images')
    parser.add_argument('--train-ann', default='coco/COCO_amodal_train2014.json',
                        help='COCOA training annotations')
    parser.add_argument('--val-ann', default='coco/COCO_amodal_val2014.json',
                        help='COCOA validation annotations')
    parser.add_argument('--min-occlusion', type=float, default=0.00,
                        help='Minimum occlusion rate (0.05=slightly occluded)')
    parser.add_argument('--min-area', type=int, default=400,
                        help='Minimum box area in pixels')
    
    # Model
    parser.add_argument('--dfine-config', default='models/dfine_hgnetv2_x_obj2coco.yml',
                        help='DFINE config file')
    parser.add_argument('--dfine-checkpoint', default='models/dfine_0.73.pth',
                        help='DFINE checkpoint (your segmentation model)')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension for amodal head')
    parser.add_argument('--roi-size', type=int, default=7,
                        help='RoI feature size')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Dataloader workers')
    parser.add_argument('--output-dir', default='outputs/amodal_humans',
                        help='Output directory')
    
    return parser.parse_args()


def train_epoch(amodal_head, backbone, train_loader, criterion, optimizer, device, epoch, roi_size):
    """Train one epoch"""
    amodal_head.train()
    backbone.eval()
    
    total_loss = 0
    metrics = {'offset': 0, 'giou': 0, 'occlusion': 0, 'mean_giou': 0, 'mean_iou': 0}
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        images = batch['image'].to(device)
        visible_boxes = batch['visible_boxes'].to(device)
        amodal_boxes = batch['amodal_boxes'].to(device)
        occlusion_scores = batch['occlusion_scores'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Extract backbone features (frozen)
        with torch.no_grad():
            features = backbone(images)
            if isinstance(features, (list, tuple)):
                feature_map = features[-1]  # Use last layer
            else:
                feature_map = features
        
        batch_loss = 0
        batch_metrics = {k: 0 for k in metrics}
        valid_samples = 0
        
        # Process each image in batch
        for i in range(images.size(0)):
            mask_i = valid_mask[i] > 0
            if mask_i.sum() == 0:
                continue
            
            visible_boxes_i = visible_boxes[i][mask_i]
            amodal_boxes_i = amodal_boxes[i][mask_i]
            occlusion_i = occlusion_scores[i][mask_i]
            
            # Extract RoI features
            feature_map_i = feature_map[i:i+1]
            roi_features = extract_roi_features(feature_map_i, visible_boxes_i, roi_size)
            
            # Predict amodal boxes
            predictions = amodal_head(roi_features, visible_boxes_i)
            
            targets = {
                'amodal_boxes': amodal_boxes_i,
                'occlusion_scores': occlusion_i
            }
            
            # Compute loss
            loss, loss_dict = criterion(predictions, targets)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                batch_loss += loss
                for key in metrics:
                    if key in loss_dict:
                        batch_metrics[key] += loss_dict[key]
                    elif f'loss_{key}' in loss_dict:
                        batch_metrics[key] += loss_dict[f'loss_{key}']
                valid_samples += 1
        
        if valid_samples > 0:
            batch_loss = batch_loss / valid_samples
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(amodal_head.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += batch_loss.item()
            for key in metrics:
                metrics[key] += batch_metrics[key] / valid_samples
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'giou': f'{batch_metrics["mean_giou"]/valid_samples:.3f}',
                'iou': f'{batch_metrics["mean_iou"]/valid_samples:.3f}'
            })
    
    if num_batches == 0:
        return 0, metrics
    
    return total_loss / num_batches, {k: v / num_batches for k, v in metrics.items()}


def validate(amodal_head, backbone, val_loader, criterion, device, roi_size):
    """Validate"""
    amodal_head.eval()
    backbone.eval()
    
    total_loss = 0
    metrics = {'offset': 0, 'giou': 0, 'occlusion': 0, 'mean_giou': 0, 'mean_iou': 0}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            visible_boxes = batch['visible_boxes'].to(device)
            amodal_boxes = batch['amodal_boxes'].to(device)
            occlusion_scores = batch['occlusion_scores'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            # Extract features
            features = backbone(images)
            if isinstance(features, (list, tuple)):
                feature_map = features[-1]
            else:
                feature_map = features
            
            batch_loss = 0
            batch_metrics = {k: 0 for k in metrics}
            valid_samples = 0
            
            for i in range(images.size(0)):
                mask_i = valid_mask[i] > 0
                if mask_i.sum() == 0:
                    continue
                
                visible_boxes_i = visible_boxes[i][mask_i]
                amodal_boxes_i = amodal_boxes[i][mask_i]
                occlusion_i = occlusion_scores[i][mask_i]
                
                feature_map_i = feature_map[i:i+1]
                roi_features = extract_roi_features(feature_map_i, visible_boxes_i, roi_size)
                
                predictions = amodal_head(roi_features, visible_boxes_i)
                
                targets = {
                    'amodal_boxes': amodal_boxes_i,
                    'occlusion_scores': occlusion_i
                }
                
                loss, loss_dict = criterion(predictions, targets)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    batch_loss += loss
                    for key in metrics:
                        if key in loss_dict:
                            batch_metrics[key] += loss_dict[key]
                        elif f'loss_{key}' in loss_dict:
                            batch_metrics[key] += loss_dict[f'loss_{key}']
                    valid_samples += 1
            
            if valid_samples > 0:
                total_loss += (batch_loss / valid_samples).item()
                for key in metrics:
                    metrics[key] += batch_metrics[key] / valid_samples
                num_batches += 1
    
    if num_batches == 0:
        return 0, metrics
    
    return total_loss / num_batches, {k: v / num_batches for k, v in metrics.items()}


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("🎯 TRAINING AMODAL DETECTION HEAD - OCCLUDED HUMANS")
    print("="*80)
    print(f"Strategy: Freeze DFINE backbone + train amodal offset head")
    print(f"Dataset: COCOA (humans only, occlusion >= {args.min_occlusion:.2f})")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load DFINE model
    print("📦 Loading DFINE model...")
    dfine_model = load_dfine_model(args.dfine_config, args.dfine_checkpoint, device)
    
    # Freeze DFINE (backbone + segmentation head)
    print("❄️  Freezing DFINE backbone and segmentation head...")
    for param in dfine_model.parameters():
        param.requires_grad = False
    dfine_model.eval()
    
    # Get backbone
    backbone = dfine_model.backbone
    backbone_channels = get_backbone_channels(dfine_model, device)
    print(f"   Backbone channels: {backbone_channels}")
    print(f"   Using layer: {backbone_channels[-1]} channels\n")
    
    # Create amodal head
    print("🏗️  Creating amodal offset head...")
    amodal_head = AmodalOffsetHead(
        in_channels=backbone_channels[-1],
        hidden_dim=args.hidden_dim,
        roi_size=args.roi_size
    ).to(device)
    
    trainable_params = sum(p.numel() for p in amodal_head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in dfine_model.parameters())
    print(f"   Amodal head params: {trainable_params:,}")
    print(f"   Total DFINE params: {total_params:,} (frozen)")
    print(f"   Training: {trainable_params:,} params\n")
    
    # Create datasets
    print("📂 Loading datasets...")
    train_dataset = COCOAAmodalDataset(
        image_dir=args.train_images,
        ann_file=args.train_ann,
        split='train',
        image_size=640,
        max_objects=50,
        augment=True,
        min_occlusion=args.min_occlusion,
        min_area=args.min_area
    )
    
    val_dataset = COCOAAmodalDataset(
        image_dir=args.val_images,
        ann_file=args.val_ann,
        split='val',
        image_size=640,
        max_objects=50,
        augment=False,
        min_occlusion=args.min_occlusion,
        min_area=args.min_area
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Optimizer and loss
    criterion = AmodalLoss(
        weight_offset=10.0,
        weight_giou=5.0,
        weight_occlusion=2.0
    )
    
    optimizer = optim.AdamW(
        amodal_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...\n")
    best_giou = 0
    best_iou = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"{'='*80}")
        print(f"📅 Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        train_loss, train_metrics = train_epoch(
            amodal_head, backbone, train_loader, criterion, optimizer, device, epoch, args.roi_size
        )
        
        val_loss, val_metrics = validate(
            amodal_head, backbone, val_loader, criterion, device, args.roi_size
        )
        
        scheduler.step()
        
        print(f"\n📊 Results:")
        print(f"   Train: loss={train_loss:.4f}, GIoU={train_metrics['mean_giou']:.3f}, IoU={train_metrics['mean_iou']:.3f}")
        print(f"   Val:   loss={val_loss:.4f}, GIoU={val_metrics['mean_giou']:.3f}, IoU={val_metrics['mean_iou']:.3f}")
        
        # Save best model
        if val_metrics['mean_giou'] > best_giou:
            best_giou = val_metrics['mean_giou']
            best_iou = val_metrics['mean_iou']
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': amodal_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_giou': best_giou,
                'best_iou': best_iou,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
            print(f"   🏆 New best! GIoU: {best_giou:.3f}, IoU: {best_iou:.3f}")
        
        # Save checkpoints
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': amodal_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_giou': best_giou,
                'args': vars(args)
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        print()
    
    print("="*80)
    print("🎉 Training complete!")
    print(f"🏆 Best GIoU: {best_giou:.3f}, IoU: {best_iou:.3f}")
    print(f"📁 Models saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()