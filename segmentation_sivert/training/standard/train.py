#!/usr/bin/env python3
"""
Standard DFINE Segmentation Training Script
Uses the new base classes and modular structure
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append('src')

# Import core modules
from core.datasets import create_pascal_dataset
from core.losses import create_loss
from core.metrics import compute_miou, create_metrics_tracker
from core.models import load_pretrained_dfine, get_actual_backbone_channels, create_combined_model

# Import model
from models.standard import create_standard_segmentation_head


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Standard DFINE Segmentation Training')
    
    # Model configuration
    parser.add_argument('--config', default='../../base_dfine/dfine_hgnetv2_x_obj2coco.yml',
                        help='Path to DFINE configuration file')
    parser.add_argument('--checkpoint', default='../../base_dfine/dfine_x_obj2coco.pth',
                        help='Path to pretrained DFINE checkpoint')
    
    # Dataset configuration
    parser.add_argument('--dataset', default='../../datasets/pascal_person_parts',
                        help='Path to Pascal Person Parts dataset')
    parser.add_argument('--num-classes', type=int, default=7,
                        help='Number of segmentation classes')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Model configuration
    parser.add_argument('--image-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--feature-dim', type=int, default=256,
                        help='Feature dimension for segmentation head')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training settings
    parser.add_argument('--multi-scale', action='store_true',
                        help='Enable multi-scale training')
    parser.add_argument('--freeze-detection', action='store_true', default=True,
                        help='Freeze detection weights during training')
    parser.add_argument('--unfreeze-epoch', type=int, default=40,
                        help='Epoch to start unfreezing backbone')
    
    # System configuration
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    
    # Output configuration
    parser.add_argument('--output-dir', default='../../outputs/standard_segmentation',
                        help='Output directory for saving models')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Logging
    parser.add_argument('--wandb-project', default='dfine-standard-segmentation',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-run-name', default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup compute device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"🚀 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_model(args, device):
    """Create the segmentation model"""
    print("🏗️  Creating standard segmentation model...")
    
    # Load pretrained DFINE model
    dfine_model = load_pretrained_dfine(args.config, args.checkpoint)
    
    # Get backbone channels
    backbone_channels = get_actual_backbone_channels(dfine_model)
    print(f"   Detected backbone channels: {backbone_channels}")
    
    # Create standard segmentation head
    seg_head = create_standard_segmentation_head(
        in_channels_list=backbone_channels,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        dropout_rate=args.dropout_rate
    )
    
    # Create combined model
    model = create_combined_model(
        dfine_model=dfine_model,
        seg_head=seg_head,
        freeze_detection=args.freeze_detection
    )
    
    model = model.to(device)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"📊 Model Information:")
    print(f"   Tier: {model_info['tier']}")
    print(f"   Total parameters: {model_info['total_params']:,}")
    print(f"   Trainable parameters: {model_info['trainable_params']:,}")
    print(f"   Model size: {model_info['model_size_mb']:.2f} MB")
    
    return model


def create_datasets(args):
    """Create training and validation datasets"""
    print("📊 Creating datasets...")
    
    # Training dataset with augmentations
    train_dataset = create_pascal_dataset(
        root_dir=args.dataset,
        split='train',
        image_size=args.image_size,
        tier='standard',
        multi_scale=args.multi_scale
    )
    
    # Validation dataset without augmentations
    val_dataset = create_pascal_dataset(
        root_dir=args.dataset,
        split='val',
        image_size=args.image_size,
        tier='standard',
        multi_scale=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def create_optimizer_and_scheduler(model, args):
    """Create optimizer and learning rate scheduler"""
    
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Create optimizer
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,
        T_mult=2,
        eta_min=1e-6
    )
    
    print(f"🎯 Optimizer: AdamW with {len(trainable_params)} parameter groups")
    print(f"📈 Scheduler: CosineAnnealingWarmRestarts")
    
    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_miou = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        seg_pred = outputs['segmentation']
        
        # Calculate loss
        loss, loss_dict = criterion(seg_pred, masks)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            miou = compute_miou(seg_pred, masks)
        
        # Update running averages
        total_loss += loss.item()
        total_miou += miou
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'mIoU': f'{miou:.3f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # Log batch metrics
        if batch_idx % 20 == 0 and not args.no_wandb:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_miou': miou,
                'train/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })
    
    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    
    return avg_loss, avg_miou


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0.0
    total_miou = 0.0
    num_batches = len(val_loader)
    
    # Create metrics tracker for detailed evaluation
    metrics_tracker = create_metrics_tracker(num_classes=7)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            seg_pred = outputs['segmentation']
            
            # Calculate loss
            loss, loss_dict = criterion(seg_pred, masks)
            
            # Calculate metrics
            miou = compute_miou(seg_pred, masks)
            
            # Update metrics tracker
            metrics_tracker.update(seg_pred, masks)
            
            # Update running averages
            total_loss += loss.item()
            total_miou += miou
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'mIoU': f'{miou:.3f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    
    # Get detailed metrics
    detailed_metrics = metrics_tracker.get_summary()
    
    return avg_loss, avg_miou, detailed_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, best_miou, args, filename=None):
    """Save training checkpoint"""
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    filepath = os.path.join(args.output_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_miou': best_miou,
        'args': vars(args),
        'model_info': model.get_model_info()
    }, filepath)
    
    print(f"💾 Saved checkpoint: {filepath}")


def main():
    """Main training function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = setup_device(args.device)
    
    # Initialize wandb
    if not args.no_wandb:
        run_name = args.wandb_run_name or f'standard_seg_{args.epochs}ep'
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
    
    # Create model
    model = create_model(args, device)
    
    # Create datasets
    train_loader, val_loader = create_datasets(args)
    
    # Create loss function
    criterion = create_loss('standard', num_classes=args.num_classes)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args)
    
    # Training loop
    best_miou = 0.0
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"📁 Output directory: {args.output_dir}")
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.epochs}")
        
        # Unfreeze backbone if specified
        if epoch == args.unfreeze_epoch and args.freeze_detection:
            print("🔓 Unfreezing detection backbone")
            model.unfreeze_detection()
            
            # Add backbone parameters to optimizer
            backbone_params = [p for p in model.dfine_model.backbone.parameters() if p.requires_grad]
            if backbone_params:
                optimizer.add_param_group({
                    'params': backbone_params,
                    'lr': args.lr * 0.1  # Lower LR for backbone
                })
        
        # Train
        train_loss, train_miou = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args
        )
        
        # Validate
        val_loss, val_miou, detailed_metrics = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch results
        print(f"📊 Train - Loss: {train_loss:.4f}, mIoU: {train_miou:.3f}")
        print(f"📊 Val   - Loss: {val_loss:.4f}, mIoU: {val_miou:.3f}")
        
        # Log to wandb
        if not args.no_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/miou': train_miou,
                'val/loss': val_loss,
                'val/miou': val_miou,
                'lr': optimizer.param_groups[0]['lr']
            }
            
            # Add detailed metrics
            for key, value in detailed_metrics.items():
                log_dict[f'val/{key}'] = value
            
            wandb.log(log_dict)
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(model, optimizer, scheduler, epoch, best_miou, args, 'best_model.pth')
            print(f"🏆 New best model! mIoU: {best_miou:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_miou, args)
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation mIoU: {best_miou:.4f}")
    print(f"📁 Models saved in: {args.output_dir}")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()