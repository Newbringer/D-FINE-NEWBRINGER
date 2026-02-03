#!/usr/bin/env python3
"""
Train Amodal Offset Head for HUMANS ONLY
Simple, clean, focused on amodal detection
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
sys.path.insert(0, str(PROJECT_ROOT / 'segmentation_sivert'))
sys.path.insert(0, str(PROJECT_ROOT / 'glass_wall_detection' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'amodal_detection_head'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

from cocoa_dataset import COCOAAmodalDataset, collate_fn
from amodal_head import AmodalOffsetHead, AmodalOffsetLoss, extract_roi_features

try:
    from segmentation_sivert.core.models import load_pretrained_dfine
except ImportError:
    # Fallback if path is different
    try:
        from src.core import YAMLConfig
    except ImportError:
        from core import YAMLConfig
    
    def load_pretrained_dfine(config_path, checkpoint_path):
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
        return model


def get_backbone_channels(model, device='cpu'):
    """Get actual backbone output channels"""
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 640, 640).to(device)
        features = model.backbone(dummy)
        channels = [f.shape[1] for f in features]
    return channels


def parse_args():
    parser = argparse.ArgumentParser(description='Train Amodal Detection for Humans')
    
    # Data
    parser.add_argument('--train-images', default='coco/train2014')
    parser.add_argument('--val-images', default='coco/val2014')
    parser.add_argument('--train-official', default='coco/COCO_amodal_train2014.json',
                        help='Official COCOA (has human categories)')
    parser.add_argument('--train-detectron', default='coco/COCO_amodal_train2014_detectron.json',
                        help='Detectron COCOA (backup for extra data)')
    parser.add_argument('--val-official', default='coco/COCO_amodal_val2014.json')
    parser.add_argument('--val-detectron', default='coco/COCO_amodal_val2014_detectron.json')
    parser.add_argument('--min-occlusion', type=float, default=0.05,
                        help='0.0=all, 0.05=slightly occluded, 0.1=clearly occluded (RECOMMENDED)')
    
    # Model
    parser.add_argument('--dfine-config', default='models/dfine_hgnetv2_x_obj2coco.yml')
    parser.add_argument('--dfine-checkpoint', default='models/dfine_0.73.pth')
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--roi-size', type=int, default=7)
    
    # Training
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir', default='outputs/amodal_humans')
    
    return parser.parse_args()


def create_dataloaders(args):
    """Create dataloaders"""
    train_dataset = COCOAAmodalDataset(
        image_dir=args.train_images,
        official_ann_file=args.train_official,
        detectron_ann_file=args.train_detectron,
        split='train',
        image_size=640,
        max_objects=50,
        augment=True,
        min_occlusion=args.min_occlusion,
        min_area=400
    )
    
    val_dataset = COCOAAmodalDataset(
        image_dir=args.val_images,
        official_ann_file=args.val_official,
        detectron_ann_file=args.val_detectron,
        split='val',
        image_size=640,
        max_objects=50,
        augment=False,
        min_occlusion=args.min_occlusion,
        min_area=400
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
    
    return train_loader, val_loader


def train_epoch(model, dfine_backbone, loader, criterion, optimizer, device, epoch, roi_size=7):
    """Train one epoch"""
    model.train()
    dfine_backbone.eval()
    
    total_loss = 0
    metrics = {'offset': 0, 'giou': 0, 'occlusion': 0, 'mean_giou': 0}
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        images = batch['image'].to(device)
        visible_boxes_gt = batch['visible_boxes'].to(device)
        amodal_boxes_gt = batch['amodal_boxes'].to(device)
        occlusion_scores_gt = batch['occlusion_scores'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Get DFINE backbone features (frozen)
        with torch.no_grad():
            backbone_features = dfine_backbone(images)
            feature_map = backbone_features[-1]
        
        batch_loss = 0
        batch_metrics = {'offset': 0, 'giou': 0, 'occlusion': 0, 'mean_giou': 0}
        valid_samples = 0
        
        # Process each image
        for i in range(len(images)):
            mask_i = valid_mask[i] > 0
            if mask_i.sum() == 0:
                continue
            
            visible_boxes_i = visible_boxes_gt[i][mask_i]
            amodal_boxes_i = amodal_boxes_gt[i][mask_i]
            occlusion_i = occlusion_scores_gt[i][mask_i]
            
            # Extract RoI features
            feature_map_i = feature_map[i:i+1]
            roi_features = extract_roi_features(
                feature_map_i,
                visible_boxes_i,
                roi_size=roi_size
            )
            
            # Predict amodal offset
            predictions = model(roi_features, visible_boxes_i)
            
            targets = {
                'amodal_boxes': amodal_boxes_i,
                'occlusion_scores': occlusion_i
            }
            
            loss, loss_dict = criterion(predictions, targets)
            
            if not torch.isnan(loss):
                batch_loss += loss
                for key in metrics:
                    if f'loss_{key}' in loss_dict:
                        batch_metrics[key] += loss_dict[f'loss_{key}']
                    elif key in loss_dict:
                        batch_metrics[key] += loss_dict[key]
                valid_samples += 1
        
        if valid_samples > 0:
            batch_loss = batch_loss / valid_samples
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += batch_loss.item()
            for key in metrics:
                metrics[key] += batch_metrics[key] / valid_samples
            
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'giou': f'{batch_metrics["mean_giou"]/valid_samples:.3f}'
            })
    
    num_batches = len(loader)
    return total_loss / num_batches, {k: v / num_batches for k, v in metrics.items()}


def validate(model, dfine_backbone, loader, criterion, device, roi_size=7):
    """Validate"""
    model.eval()
    dfine_backbone.eval()
    
    total_loss = 0
    metrics = {'offset': 0, 'giou': 0, 'occlusion': 0, 'mean_giou': 0}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            images = batch['image'].to(device)
            visible_boxes_gt = batch['visible_boxes'].to(device)
            amodal_boxes_gt = batch['amodal_boxes'].to(device)
            occlusion_scores_gt = batch['occlusion_scores'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            backbone_features = dfine_backbone(images)
            feature_map = backbone_features[-1]
            
            batch_loss = 0
            batch_metrics = {'offset': 0, 'giou': 0, 'occlusion': 0, 'mean_giou': 0}
            valid_samples = 0
            
            for i in range(len(images)):
                mask_i = valid_mask[i] > 0
                if mask_i.sum() == 0:
                    continue
                
                visible_boxes_i = visible_boxes_gt[i][mask_i]
                amodal_boxes_i = amodal_boxes_gt[i][mask_i]
                occlusion_i = occlusion_scores_gt[i][mask_i]
                
                feature_map_i = feature_map[i:i+1]
                roi_features = extract_roi_features(
                    feature_map_i,
                    visible_boxes_i,
                    roi_size=roi_size
                )
                
                predictions = model(roi_features, visible_boxes_i)
                
                targets = {
                    'amodal_boxes': amodal_boxes_i,
                    'occlusion_scores': occlusion_i
                }
                
                loss, loss_dict = criterion(predictions, targets)
                
                if not torch.isnan(loss):
                    batch_loss += loss
                    for key in metrics:
                        if f'loss_{key}' in loss_dict:
                            batch_metrics[key] += loss_dict[f'loss_{key}']
                        elif key in loss_dict:
                            batch_metrics[key] += loss_dict[key]
                    valid_samples += 1
            
            if valid_samples > 0:
                total_loss += (batch_loss / valid_samples).item()
                for key in metrics:
                    metrics[key] += batch_metrics[key] / valid_samples
    
    num_batches = len(loader)
    return total_loss / num_batches, {k: v / num_batches for k, v in metrics.items()}


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("🎯 TRAINING AMODAL DETECTION - HUMANS ONLY")
    print("="*80)
    print("Strategy: Use DFINE's detection + predict amodal offset")
    print("Dataset: Official COCOA + Detectron backup")
    print(f"   Prioritizes Official (has human categories)")
    print(f"   Uses Detectron for images without humans in Official")
    print(f"Min occlusion: {args.min_occlusion:.2f}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load DFINE backbone (frozen)
    print("\n📦 Loading DFINE backbone...")
    dfine_model = load_pretrained_dfine(args.dfine_config, args.dfine_checkpoint)
    dfine_model = dfine_model.to(device)  # Move entire model to device first
    dfine_backbone = dfine_model.backbone
    dfine_backbone.eval()
    for param in dfine_backbone.parameters():
        param.requires_grad = False
    
    backbone_channels = get_backbone_channels(dfine_model, device)
    print(f"   Backbone channels: {backbone_channels}")
    print(f"   Using final layer: {backbone_channels[-1]} channels")
    
    # Create amodal offset head
    print(f"\n🏗️  Creating amodal offset head...")
    model = AmodalOffsetHead(
        roi_size=args.roi_size,
        in_channels=backbone_channels[-1],
        hidden_dim=args.hidden_dim
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    
    # Optimizer and loss
    criterion = AmodalOffsetLoss(
        weight_offset=10.0,
        weight_giou=5.0,
        weight_occlusion=2.0
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    best_giou = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n📅 Epoch {epoch}/{args.epochs}")
        
        train_loss, train_metrics = train_epoch(
            model, dfine_backbone, train_loader, criterion, optimizer, device, epoch, args.roi_size
        )
        
        val_loss, val_metrics = validate(
            model, dfine_backbone, val_loader, criterion, device, args.roi_size
        )
        
        scheduler.step()
        
        print(f"📊 Train: loss={train_loss:.4f}, GIoU={train_metrics['mean_giou']:.3f}")
        print(f"📊 Val: loss={val_loss:.4f}, GIoU={val_metrics['mean_giou']:.3f}")
        
        # Save best model by GIoU
        if val_metrics['mean_giou'] > best_giou:
            best_giou = val_metrics['mean_giou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_giou': best_giou,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"🏆 New best! GIoU: {best_giou:.3f}")
        
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_giou': best_giou,
                'args': vars(args)
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    print(f"\n🎉 Training complete!")
    print(f"🏆 Best GIoU: {best_giou:.3f}")
    print(f"📁 Models saved to: {args.output_dir}")


if __name__ == '__main__':
    main()