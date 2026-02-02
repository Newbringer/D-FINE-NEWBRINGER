#!/usr/bin/env python3
"""
Training Script for Amodal Person Detection using COCOA Dataset
Perfect for DFINE since it's already pretrained on COCO!
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'segmentation_sivert'))
sys.path.insert(0, str(PROJECT_ROOT / 'glass_wall_detection' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'amodal_detection_head'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

from cocoa_dataset import COCOAAmodalDataset, collate_fn
from amodal_head import (
    AmodalDetectionHead,
    AmodalLoss,
    CombinedDFINEAmodalModel
)
from segmentation_sivert.core.models import load_pretrained_dfine, get_actual_backbone_channels
from model_architecture import SegmentationHead


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Amodal Person Detection on COCOA')
    
    # Paths
    parser.add_argument('--coco-train-root', 
                        default='coco/train2014',
                        help='Path to COCO train2014 images directory')
    parser.add_argument('--coco-val-root',
                        default='coco/val2014',
                        help='Path to COCO val2014 images directory')
    parser.add_argument('--cocoa-train-ann', 
                        default='coco/COCO_amodal_train2014_detectron.json',
                        help='Path to COCOA training annotation JSON')
    parser.add_argument('--cocoa-val-ann', 
                        default='coco/COCO_amodal_val2014_detectron.json',
                        help='Path to COCOA validation annotation JSON')
    parser.add_argument('--dfine-config', default='models/dfine_hgnetv2_x_obj2coco.yml',
                        help='Path to DFINE config')
    parser.add_argument('--dfine-checkpoint', default='models/dfine_0.73.pth',
                        help='Path to DFINE checkpoint')
    parser.add_argument('--seg-checkpoint', default='models/dfine_0.73.pth',
                        help='Path to trained segmentation model checkpoint')
    parser.add_argument('--output-dir', default='outputs/amodal_cocoa_person',
                        help='Output directory for checkpoints')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--image-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--max-objects', type=int, default=50,
                        help='Maximum objects per image')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension for amodal head')
    parser.add_argument('--num-queries', type=int, default=100,
                        help='Number of detection queries (100 for person-only)')
    
    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training')
    
    # Logging
    parser.add_argument('--wandb-project', default='dfine-amodal-cocoa-person',
                        help='Weights & Biases project name')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def load_existing_models(args):
    """Load pretrained DFINE and segmentation models"""
    print("🚀 Loading pretrained models...")
    
    # Load DFINE
    print(f"📦 Loading DFINE from {args.dfine_checkpoint}")
    dfine_model = load_pretrained_dfine(args.dfine_config, args.dfine_checkpoint)
    
    # Get backbone channels
    backbone_channels = get_actual_backbone_channels(dfine_model)
    print(f"   Backbone channels: {backbone_channels}")
    
    # Load segmentation checkpoint
    print(f"📦 Loading segmentation from {args.seg_checkpoint}")
    seg_checkpoint = torch.load(args.seg_checkpoint, map_location='cpu')
    
    # Extract hyperparameters
    if 'hyperparameters' in seg_checkpoint:
        hyper = seg_checkpoint['hyperparameters']
        feature_dim = hyper.get('feature_dim', 256)
        num_classes = hyper.get('num_classes', 7)
    else:
        feature_dim = 256
        num_classes = 7
    
    print(f"   Feature dim: {feature_dim}")
    print(f"   Num classes: {num_classes}")
    
    # Create segmentation head
    segmentation_head = SegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=num_classes,
        feature_dim=feature_dim,
        dropout_rate=0.1
    )
    
    # Load segmentation weights
    if isinstance(seg_checkpoint, dict) and 'model_state_dict' in seg_checkpoint:
        state_dict = seg_checkpoint['model_state_dict']
    elif isinstance(seg_checkpoint, dict) and 'state_dict' in seg_checkpoint:
        state_dict = seg_checkpoint['state_dict']
    else:
        state_dict = seg_checkpoint

    seg_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('seg_head.'):
            new_key = key.replace('seg_head.', '')
            seg_state_dict[new_key] = value
        elif key.startswith('segmentation_head.'):
            new_key = key.replace('segmentation_head.', '')
            seg_state_dict[new_key] = value

    if seg_state_dict:
        model_state = segmentation_head.state_dict()
        filtered_state = {
            k: v for k, v in seg_state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        
        if filtered_state:
            missing, unexpected = segmentation_head.load_state_dict(filtered_state, strict=False)
            print("✅ Loaded segmentation weights")
            if missing:
                print(f"   Missing keys: {len(missing)}")
            if unexpected:
                print(f"   Unexpected keys: {len(unexpected)}")
        else:
            print("⚠️  No compatible segmentation weights found. Using random init.")
    
    return dfine_model, segmentation_head, backbone_channels


def create_dataloaders(args):
    """Create train and val dataloaders"""
    print("📊 Creating COCOA person dataloaders...")
    
    train_dataset = COCOAAmodalDataset(
        coco_root=args.coco_train_root,
        cocoa_annotation_file=args.cocoa_train_ann,
        split='train',
        image_size=args.image_size,
        max_objects=args.max_objects,
        augment=True,
        person_only=True
    )
    
    val_dataset = COCOAAmodalDataset(
        coco_root=args.coco_val_root,
        cocoa_annotation_file=args.cocoa_val_ann,
        split='val',
        image_size=args.image_size,
        max_objects=args.max_objects,
        augment=False,
        person_only=True
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
    
    print(f"   Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"   Val: {len(val_dataset)} images, {len(val_loader)} batches")
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    loss_components = {
        'visible_l1': 0.0,
        'amodal_l1': 0.0,
        'visible_giou': 0.0,
        'amodal_giou': 0.0,
        'class': 0.0,
        'occlusion': 0.0
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        
        targets = {
            'visible_boxes': batch['visible_boxes'].to(device),
            'amodal_boxes': batch['amodal_boxes'].to(device),
            'class_labels': batch['class_labels'].to(device),
            'occlusion_scores': batch['occlusion_scores'].to(device),
            'valid_mask': batch['valid_mask'].to(device)
        }
        
        optimizer.zero_grad()
        outputs = model(images)
        
        predictions = {
            'visible_boxes': outputs['visible_boxes'],
            'amodal_boxes': outputs['amodal_boxes'],
            'occlusion_scores': outputs['occlusion_scores'],
            'class_logits': outputs['class_logits'],
            'confidence_scores': outputs['confidence_scores']
        }
        
        loss, loss_dict = criterion(predictions, targets)
        
        if torch.isnan(loss):
            print(f"⚠️  NaN loss at batch {batch_idx}, skipping...")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for key in loss_components:
            if f'loss_{key}' in loss_dict:
                loss_components[key] += loss_dict[f'loss_{key}']
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'vis_l1': f'{loss_dict["loss_visible_l1"]:.4f}',
            'amod_l1': f'{loss_dict["loss_amodal_l1"]:.4f}'
        })
    
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return avg_loss, loss_components


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0.0
    loss_components = {
        'visible_l1': 0.0,
        'amodal_l1': 0.0,
        'visible_giou': 0.0,
        'amodal_giou': 0.0,
        'class': 0.0,
        'occlusion': 0.0
    }
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for batch in pbar:
            images = batch['image'].to(device)
            
            targets = {
                'visible_boxes': batch['visible_boxes'].to(device),
                'amodal_boxes': batch['amodal_boxes'].to(device),
                'class_labels': batch['class_labels'].to(device),
                'occlusion_scores': batch['occlusion_scores'].to(device),
                'valid_mask': batch['valid_mask'].to(device)
            }
            
            outputs = model(images)
            
            predictions = {
                'visible_boxes': outputs['visible_boxes'],
                'amodal_boxes': outputs['amodal_boxes'],
                'occlusion_scores': outputs['occlusion_scores'],
                'class_logits': outputs['class_logits'],
                'confidence_scores': outputs['confidence_scores']
            }
            
            loss, loss_dict = criterion(predictions, targets)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                for key in loss_components:
                    if f'loss_{key}' in loss_dict:
                        loss_components[key] += loss_dict[f'loss_{key}']
    
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return avg_loss, loss_components


def save_checkpoint(model, optimizer, epoch, best_loss, args, filename=None):
    """Save training checkpoint"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    if filename is None:
        filename = f'amodal_cocoa_person_epoch_{epoch}.pth'
    
    filepath = os.path.join(args.output_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'amodal_head_state_dict': model.amodal_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'args': vars(args)
    }, filepath)
    
    print(f"💾 Saved checkpoint: {filepath}")


def main():
    """Main training function"""
    args = parse_args()
    
    print("=" * 80)
    print("🎯 TRAINING AMODAL PERSON DETECTION ON COCOA DATASET")
    print("=" * 80)
    print(f"📋 Configuration:")
    print(f"   COCO train images: {args.coco_train_root}")
    print(f"   COCO val images: {args.coco_val_root}")
    print(f"   COCOA train ann: {args.cocoa_train_ann}")
    print(f"   COCOA val ann: {args.cocoa_val_ann}")
    print(f"   Segmentation checkpoint: {args.seg_checkpoint}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Num queries: {args.num_queries} (person-only)")
    print(f"   ✨ PERFECT MATCH: DFINE pretrained on COCO + Amodal person annotations!")
    print("=" * 80)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f'cocoa_person_{args.epochs}ep_bs{args.batch_size}'
        )
    
    # Load existing models
    dfine_model, segmentation_head, backbone_channels = load_existing_models(args)
    
    # Create amodal head
    print("🏗️  Creating amodal detection head...")
    amodal_head = AmodalDetectionHead(
        in_channels=backbone_channels[-1],
        hidden_dim=args.hidden_dim,
        num_classes=2,  # Background + person only
        num_queries=args.num_queries
    )
    
    # Create combined model
    model = CombinedDFINEAmodalModel(
        dfine_model=dfine_model,
        segmentation_head=segmentation_head,
        amodal_head=amodal_head
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable (amodal head only): {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    
    # Create loss and optimizer
    criterion = AmodalLoss()
    optimizer = optim.AdamW(
        model.amodal_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"   Training on PERSON CLASS ONLY from COCOA dataset")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n📅 Epoch {epoch}/{args.epochs}")
        
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        val_loss, val_components = validate_epoch(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        # Print results
        print(f"📊 Train Loss: {train_loss:.4f}")
        print(f"   Visible L1: {train_components['visible_l1']:.4f}")
        print(f"   Amodal L1: {train_components['amodal_l1']:.4f}")
        print(f"   Visible GIoU: {train_components['visible_giou']:.4f}")
        print(f"   Amodal GIoU: {train_components['amodal_giou']:.4f}")
        
        print(f"📊 Val Loss: {val_loss:.4f}")
        print(f"   Visible L1: {val_components['visible_l1']:.4f}")
        print(f"   Amodal L1: {val_components['amodal_l1']:.4f}")
        
        if not args.no_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'lr': optimizer.param_groups[0]['lr']
            }
            for key, value in train_components.items():
                log_dict[f'train/{key}'] = value
            for key, value in val_components.items():
                log_dict[f'val/{key}'] = value
            
            wandb.log(log_dict)
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_loss, args, 'best_cocoa_person.pth')
            print(f"🏆 New best model! Val loss: {best_loss:.4f}")
        
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, best_loss, args)
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation loss: {best_loss:.4f}")
    print(f"📁 Models saved in: {args.output_dir}")
    print(f"🎯 Model trained on PERSON CLASS ONLY")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    # Example usage:
    # python amodal_detection_head/train_amodal.py \
    #     --seg-checkpoint path/to/segmentation.pth \
    #     --batch-size 8 \
    #     --epochs 50 \
    #     --lr 1e-4
    #
    # Or with custom paths:
    # python amodal_detection_head/train_amodal.py \
    #     --coco-train-root amodal_detection_head/coco/train2014 \
    #     --coco-val-root amodal_detection_head/coco/val2014 \
    #     --cocoa-train-ann amodal_detection_head/coco/COCO_amodal_train2014_detectron.json \
    #     --cocoa-val-ann amodal_detection_head/coco/COCO_amodal_val2014_detectron.json \
    #     --seg-checkpoint path/to/segmentation.pth
    main()