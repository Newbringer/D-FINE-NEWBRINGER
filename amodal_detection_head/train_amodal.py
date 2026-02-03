#!/usr/bin/env python3
"""
Train Amodal Detection Head - Synthetic Data
UPDATED: Edge-specific loss, separate train/val splits
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
sys.path.insert(0, str(PROJECT_ROOT / 'glass_wall_detection' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

from cocoa_dataset import COCOAAmodalDataset, collate_fn
from amodal_head import AmodalOffsetHead, AmodalLoss, extract_roi_features


def load_dfine_segmentation_model(config_path, checkpoint_path, device):
    """Load DFINE segmentation model"""
    print(f"\n📦 Loading DFINE segmentation model...")
    print(f"   Config: {config_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    
    from model_architecture import SegmentationHead, DFineWithSegmentation
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'ema' in checkpoint and 'module' in checkpoint['ema']:
        state_dict = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    has_seg_head = any('seg_head.' in key for key in state_dict.keys())
    
    if not has_seg_head:
        raise ValueError("Checkpoint doesn't contain segmentation head!")
    
    try:
        from src.core import YAMLConfig
    except:
        from core import YAMLConfig
    
    cfg = YAMLConfig(str(config_path))
    base_model = cfg.model
    
    base_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640)
        backbone_features = base_model.backbone(dummy_input)
        backbone_channels = [feat.shape[1] for feat in backbone_features]
    
    feature_dim = 256
    for key in state_dict.keys():
        if 'seg_head.fpn.lateral_convs.0.weight' in key:
            feature_dim = state_dict[key].shape[0]
            break
    
    seg_head = SegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=7,
        feature_dim=feature_dim,
        dropout_rate=0.1
    )
    
    model = DFineWithSegmentation(
        dfine_model=base_model,
        seg_head=seg_head,
        freeze_detection=False
    )
    
    model.load_state_dict(state_dict, strict=False)
    
    print("   ✅ Model loaded")
    return model.to(device)


def get_backbone_channels(model, device):
    """Get backbone output channels"""
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 640, 640, device=device)
        if hasattr(model, 'dfine_model'):
            features = model.dfine_model.backbone(dummy)
        else:
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
                        help='COCO training images')
    parser.add_argument('--val-images', default='coco/val2014',
                        help='COCO validation images')
    parser.add_argument('--train-ann', default='synthetic_amodal_dataset/synthetic_amodal_train/synthetic_amodal_annotations.json',
                        help='Training annotations')
    parser.add_argument('--val-ann', default='synthetic_amodal_dataset/synthetic_amodal_val/synthetic_amodal_annotations.json',
                        help='Validation annotations')
    parser.add_argument('--min-occlusion', type=float, default=0.10,
                        help='Minimum occlusion (0.10 = 10%, filters very easy cases)')
    parser.add_argument('--min-area', type=int, default=200,
                        help='Minimum box area (lowered for high-occlusion cases)')
    
    # Model
    parser.add_argument('--dfine-config', default='models/dfine_hgnetv2_x_obj2coco.yml',
                        help='DFINE config')
    parser.add_argument('--dfine-checkpoint', default='models/dfine_0.73.pth',
                        help='DFINE checkpoint')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--roi-size', type=int, default=7,
                        help='RoI size')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Workers')
    parser.add_argument('--output-dir', default='outputs/amodal_synthetic',
                        help='Output directory')
    
    return parser.parse_args()


def train_epoch(amodal_head, dfine_model, train_loader, criterion, optimizer, device, epoch, roi_size):
    """Train one epoch"""
    amodal_head.train()
    dfine_model.eval()
    
    total_loss = 0
    metrics = {'offset': 0, 'edge': 0, 'edge_x1': 0, 'edge_y1': 0, 'edge_x2': 0, 'edge_y2': 0, 
               'giou': 0, 'occlusion': 0, 'mean_giou': 0, 'mean_iou': 0}
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        images = batch['image'].to(device)
        visible_boxes = batch['visible_boxes'].to(device)
        amodal_boxes = batch['amodal_boxes'].to(device)
        occlusion_scores = batch['occlusion_scores'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            if hasattr(dfine_model, 'dfine_model'):
                features = dfine_model.dfine_model.backbone(images)
            else:
                features = dfine_model.backbone(images)
            
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
            batch_loss = batch_loss / valid_samples
            batch_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(amodal_head.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += batch_loss.item()
            for key in metrics:
                metrics[key] += batch_metrics[key] / valid_samples
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.3f}',
                'edge': f'{batch_metrics["edge"]/valid_samples:.3f}',
                'L': f'{batch_metrics["edge_x1"]/valid_samples:.2f}',
                'R': f'{batch_metrics["edge_x2"]/valid_samples:.2f}',
                'T': f'{batch_metrics["edge_y1"]/valid_samples:.2f}',
                'B': f'{batch_metrics["edge_y2"]/valid_samples:.2f}'
            })
    
    if num_batches == 0:
        return 0, metrics
    
    return total_loss / num_batches, {k: v / num_batches for k, v in metrics.items()}


def validate(amodal_head, dfine_model, val_loader, criterion, device, roi_size):
    """Validate"""
    amodal_head.eval()
    dfine_model.eval()
    
    total_loss = 0
    metrics = {'offset': 0, 'edge': 0, 'edge_x1': 0, 'edge_y1': 0, 'edge_x2': 0, 'edge_y2': 0,
               'giou': 0, 'occlusion': 0, 'mean_giou': 0, 'mean_iou': 0}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            visible_boxes = batch['visible_boxes'].to(device)
            amodal_boxes = batch['amodal_boxes'].to(device)
            occlusion_scores = batch['occlusion_scores'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            if hasattr(dfine_model, 'dfine_model'):
                features = dfine_model.dfine_model.backbone(images)
            else:
                features = dfine_model.backbone(images)
            
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
    print("🎯 AMODAL DETECTION TRAINING")
    print("="*80)
    print(f"Train: {args.train_ann}")
    print(f"Val: {args.val_ann}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load DFINE
    dfine_model = load_dfine_segmentation_model(
        args.dfine_config,
        args.dfine_checkpoint,
        device
    )
    
    # Freeze DFINE
    for param in dfine_model.parameters():
        param.requires_grad = False
    dfine_model.eval()
    
    # Get backbone channels
    backbone_channels = get_backbone_channels(dfine_model, device)
    print(f"   Backbone channels: {backbone_channels[-1]}\n")
    
    # Create amodal head
    amodal_head = AmodalOffsetHead(
        in_channels=backbone_channels[-1],
        hidden_dim=args.hidden_dim,
        roi_size=args.roi_size
    ).to(device)
    
    trainable = sum(p.numel() for p in amodal_head.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable:,}\n")
    
    # Datasets
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
    
    # Balanced loss weights
    criterion = AmodalLoss(
        weight_offset=8.0,
        weight_giou=5.0,
        weight_occlusion=2.0,
        weight_edge=20.0
    )
    
    optimizer = optim.AdamW(
        amodal_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # ReduceLROnPlateau instead of CosineAnnealing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Maximize GIoU
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
    
    # Training loop
    print(f"🚀 Training for {args.epochs} epochs...\n")
    best_giou = 0
    patience_counter = 0
    patience_limit = 7  # Stop if no improvement for 7 epochs
    
    for epoch in range(1, args.epochs + 1):
        print(f"{'='*80}")
        print(f"📅 Epoch {epoch}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*80}")
        
        train_loss, train_metrics = train_epoch(
            amodal_head, dfine_model, train_loader, criterion, optimizer, device, epoch, args.roi_size
        )
        
        val_loss, val_metrics = validate(
            amodal_head, dfine_model, val_loader, criterion, device, args.roi_size
        )
        
        # Update scheduler based on validation GIoU
        scheduler.step(val_metrics['mean_giou'])
        
        print(f"\n📊 Results:")
        print(f"   Train: loss={train_loss:.3f}, edge={train_metrics['edge']:.3f}, "
              f"L={train_metrics['edge_x1']:.2f}, R={train_metrics['edge_x2']:.2f}, "
              f"T={train_metrics['edge_y1']:.2f}, B={train_metrics['edge_y2']:.2f}")
        print(f"   Val:   loss={val_loss:.3f}, edge={val_metrics['edge']:.3f}, "
              f"GIoU={val_metrics['mean_giou']:.3f}, IoU={val_metrics['mean_iou']:.3f}")
        
        if val_metrics['mean_giou'] > best_giou:
            best_giou = val_metrics['mean_giou']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'dfine_model': dfine_model.state_dict(),
                'amodal_head': amodal_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_giou': best_giou,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
            print(f"   🏆 New best! GIoU: {best_giou:.3f}")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{patience_limit})")
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"\n⚠️  Early stopping after {epoch} epochs (no improvement for {patience_limit} epochs)")
            break
        
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'dfine_model': dfine_model.state_dict(),
                'amodal_head': amodal_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_giou': best_giou,
                'args': vars(args)
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        print()
    
    print("="*80)
    print(f"🎉 Training complete! Best GIoU: {best_giou:.3f}")
    print(f"📁 Models: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()