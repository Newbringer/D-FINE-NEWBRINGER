#!/usr/bin/env python3
"""
Train DFINE Segmentation Model with Glass Detection
Adds glass pane detection (class 80) while preserving:
- All 80 COCO classes
- Segmentation capabilities
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add DFINE to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    sys.path.insert(0, project_root)

# Import src module first to trigger all registrations
import src
from src.core import YAMLConfig

# Import dataset
from dataset import GlassWallDataset, COCORetentionDataset, MixedDataset, collate_fn


def load_segmentation_model_from_checkpoint(config_path, checkpoint_path, device):
    """Load the full segmentation model (DFINE + seg head) from checkpoint
    
    This recreates the DFineWithSegmentation architecture and loads weights.
    """
    print(f"\n📦 Loading segmentation model...")
    print(f"   Config: {config_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # Load checkpoint to inspect
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Check if this has segmentation layers
    has_seg_head = any('seg_head.' in key for key in state_dict.keys())
    has_dfine_prefix = any('dfine_model.' in key for key in state_dict.keys())
    
    print(f"   Has segmentation head: {has_seg_head}")
    print(f"   Has dfine_model prefix: {has_dfine_prefix}")
    
    if not has_seg_head:
        raise ValueError(
            "This checkpoint doesn't have a segmentation head!\n"
            "Please use a checkpoint from your segmentation training (like dfine_0.73.pth)"
        )
    
    # Load base DFINE config
    cfg = YAMLConfig(config_path)
    base_model = cfg.model
    
    # Get hyperparameters from checkpoint
    hyperparams = checkpoint.get('hyperparameters', {
        'feature_dim': 256,
        'dropout_rate': 0.1,
        'best_miou': 0.73
    })
    
    print(f"   Hyperparameters: {hyperparams}")
    
    # Recreate segmentation model architecture
    # This matches the DFineWithSegmentation from model_architecture.py
    print("   Recreating segmentation model architecture...")
    
    # We need to recreate the full model
    # Import the architecture components
    from model_architecture import SegmentationHead, DFineWithSegmentation
    
    # Get backbone channels by running a forward pass
    base_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640)
        backbone_features = base_model.backbone(dummy_input)
        backbone_channels = [feat.shape[1] for feat in backbone_features]
    
    print(f"   Backbone channels: {backbone_channels}")
    
    # Create segmentation head
    seg_head = SegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=7,  # Pascal Person Parts
        feature_dim=hyperparams.get('feature_dim', 256),
        dropout_rate=hyperparams.get('dropout_rate', 0.1)
    )
    
    # Create combined model
    model = DFineWithSegmentation(
        dfine_model=base_model,
        seg_head=seg_head,
        freeze_detection=False  # We'll handle freezing ourselves
    )
    
    # Load weights
    print("   Loading checkpoint weights...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"   ⚠️  Missing keys: {len(missing)}")
        if len(missing) <= 5:
            for key in missing:
                print(f"      - {key}")
    if unexpected:
        print(f"   ⚠️  Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 5:
            for key in unexpected:
                print(f"      - {key}")
    
    print("   ✅ Segmentation model loaded successfully")
    
    return model, cfg.criterion


def expand_to_81_classes(model):
    """Expand detection head from 80 to 81 classes for glass detection
    
    This expands:
    - All classification Linear layers (80 -> 81)
    - Denoising embeddings (81 -> 82)
    
    The new class 80 (glass_wall) is initialized with small random values.
    """
    print("\n🔧 Expanding model: 80 -> 81 classes")
    
    expanded_layers = []
    
    for name, module in model.named_modules():
        # Expand classification layers
        if any(key in name.lower() for key in ['class_embed', 'cls', 'score']):
            if isinstance(module, nn.Linear) and module.out_features == 80:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                # Create new 81-class layer
                new_module = nn.Linear(module.in_features, 81)
                
                with torch.no_grad():
                    # Copy existing 80 classes
                    new_module.weight[:80] = module.weight
                    if module.bias is not None:
                        new_module.bias[:80] = module.bias
                    
                    # Initialize new class (80) with small random values
                    nn.init.normal_(new_module.weight[80:], mean=0, std=0.01)
                    if new_module.bias is not None:
                        nn.init.zeros_(new_module.bias[80:])
                
                setattr(parent, child_name, new_module)
                expanded_layers.append(name)
                print(f"   ✅ {name}: 80 -> 81 classes")
            
            # Expand denoising embeddings
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
                expanded_layers.append(name)
                print(f"   ✅ {name}: 81 -> 82 embeddings")
    
    # Update num_classes attributes
    if hasattr(model, 'num_classes'):
        model.num_classes = 81
    if hasattr(model, 'dfine_model'):
        if hasattr(model.dfine_model, 'num_classes'):
            model.dfine_model.num_classes = 81
        if hasattr(model.dfine_model, 'decoder') and hasattr(model.dfine_model.decoder, 'num_classes'):
            model.dfine_model.decoder.num_classes = 81
    
    print(f"   Total layers expanded: {len(expanded_layers)}")
    
    return model


def freeze_for_glass_training(model, unfreeze_last_n_decoder_layers=2):
    """Freeze model strategically for efficient glass detection training
    
    Strategy:
    1. FREEZE: Entire segmentation head (keep it working as-is)
    2. FREEZE: Backbone (no need to retrain features)
    3. FREEZE: Encoder (no need to retrain multi-scale features)
    4. PARTIALLY UNFREEZE: Last N decoder layers (fine-tune for glass)
    5. UNFREEZE: Classification heads for class 80 weights
    
    This allows:
    - Efficient training (only ~1-2% of parameters)
    - Fast overfitting to glass panes
    - Preservation of COCO performance
    - Preservation of segmentation performance
    """
    print("\n🔒 Freezing model for glass detection training...")
    
    total_params = 0
    frozen_params = 0
    trainable_params = 0
    trainable_layers = []
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        should_train = False
        
        # ALWAYS FREEZE: Segmentation head
        if 'seg_head' in name:
            param.requires_grad = False
            frozen_params += param.numel()
            continue
        
        # ALWAYS FREEZE: Backbone
        if 'backbone' in name:
            param.requires_grad = False
            frozen_params += param.numel()
            continue
        
        # ALWAYS FREEZE: Encoder
        if 'encoder' in name and 'decoder' not in name:
            param.requires_grad = False
            frozen_params += param.numel()
            continue
        
        # PARTIALLY UNFREEZE: Last N decoder layers (transformer + heads)
        if unfreeze_last_n_decoder_layers > 0:
            import re
            layer_patterns = [
                r'decoder\.decoder\.layers\.(\d+)',
                r'decoder\.lqe_layers\.(\d+)',
                r'decoder\.dec_bbox_head\.(\d+)',
                r'decoder\.dec_score_head\.(\d+)',
            ]
            for pattern in layer_patterns:
                layer_match = re.search(pattern, name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    if layer_idx >= (6 - unfreeze_last_n_decoder_layers):
                        should_train = True
                        break
        
        # ALWAYS TRAIN: Classification heads (to learn class 80)
        if any(key in name for key in ['class_embed', 'score_head', 'cls']):
            should_train = True
        
        if should_train:
            param.requires_grad = True
            trainable_params += param.numel()
            trainable_layers.append(name)
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"\n   📊 Freeze summary:")
    print(f"      Total parameters: {total_params:,}")
    print(f"      Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"      Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    print(f"\n   ✅ Trainable layers ({len(trainable_layers)}):")
    for layer in trainable_layers[:15]:  # Show first 15
        print(f"      - {layer}")
    if len(trainable_layers) > 15:
        print(f"      ... and {len(trainable_layers)-15} more")
    
    return trainable_params


def train_epoch(model, criterion, dataloader, optimizer, device, epoch):
    """Train one epoch with DFINE's proper loss computation"""
    model.train()
    
    # Keep seg_head in eval mode (frozen)
    if hasattr(model, 'seg_head'):
        model.seg_head.eval()
    
    total_loss = 0
    total_glass_count = 0
    num_batches = len(dataloader)
    
    loss_components = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        
        # Move targets to device
        for target in targets:
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)
            target['image_id'] = target['image_id'].to(device)
        
        optimizer.zero_grad()
        
        try:
            # Forward pass through full model
            outputs = model(images, targets=targets)
            
            # DFINE outputs detection results, ignore segmentation
            det_outputs = {k: v for k, v in outputs.items() if k != 'segmentation'}
            
            # Compute loss using DFINE's criterion
            loss_dict = criterion(det_outputs, targets)
            
            # Sum all weighted loss components
            loss = sum(loss_dict.values())
            
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                print(f"⚠️  Invalid loss: {loss}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Track loss components
            for k, v in loss_dict.items():
                if k not in loss_components:
                    loss_components[k] = []
                loss_components[k].append(v.item())
            
            # Count glass detections
            glass_count = sum((t['labels'] == 80).sum().item() for t in targets)
            total_glass_count += glass_count
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss/(batch_idx+1):.4f}',
                'glass': glass_count
            })
        
        except Exception as e:
            print(f"\n⚠️  Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_glass = total_glass_count / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_glass, loss_components


def validate_epoch(model, criterion, dataloader, device):
    """Validate one epoch"""
    model.train()  # DFINE needs targets even during validation
    
    # Keep seg_head in eval
    if hasattr(model, 'seg_head'):
        model.seg_head.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            
            for target in targets:
                target['boxes'] = target['boxes'].to(device)
                target['labels'] = target['labels'].to(device)
                target['image_id'] = target['image_id'].to(device)
            
            try:
                outputs = model(images, targets=targets)
                det_outputs = {k: v for k, v in outputs.items() if k != 'segmentation'}
                
                loss_dict = criterion(det_outputs, targets)
                loss = sum(loss_dict.values())
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            except Exception as e:
                print(f"\n⚠️  Validation error: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir, filename='best_model.pth'):
    """Save checkpoint with hyperparameters"""
    filepath = os.path.join(output_dir, filename)
    
    # Extract just the dfine_model weights if this is a DFineWithSegmentation
    if hasattr(model, 'dfine_model') and hasattr(model, 'seg_head'):
        # Save full model for continued training
        model_state = model.state_dict()
    else:
        model_state = model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'num_classes': 81,
        'hyperparameters': {
            'feature_dim': 256,
            'dropout_rate': 0.1,
            'best_miou': 0.73,  # From original segmentation model
        }
    }, filepath)
    
    print(f"💾 Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Train Glass Detection on DFINE Segmentation Model')
    
    # Model
    parser.add_argument('--config', default='../models/dfine_hgnetv2_x_obj2coco.yml',
                       help='DFINE config file')
    parser.add_argument('--checkpoint', default='../models/dfine_0.73.pth',
                       help='Segmentation checkpoint to start from')
    
    # Data
    parser.add_argument('--glass-data', default='../data/glass_wall',
                       help='Glass wall dataset directory')
    parser.add_argument('--coco-data', default='../data/coco',
                       help='COCO dataset path for retention')
    parser.add_argument('--coco-ratio', type=float, default=1.0,
                       help='Ratio of COCO samples (0.3 = 30%% COCO, 70%% glass)')
    parser.add_argument('--coco-max-images', type=int, default=500,
                       help='Max COCO images to use')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (1e-4 recommended for fine-tuning)')
    parser.add_argument('--unfreeze-decoder-layers', type=int, default=2,
                       help='Number of decoder layers to unfreeze (0-6)')
    
    # Options
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Enable light augmentation')
    parser.add_argument('--strong-augment', action='store_true', default=True,
                       help='Enable STRONG augmentation (recommended for glass)')
    parser.add_argument('--overfit', action='store_true', default=True,
                       help='Overfit mode: train/val on same data')
    parser.add_argument('--augment-multiplier', type=int, default=3,
                       help='Repeat glass dataset N times with forced augmentation')
    
    # Output
    parser.add_argument('--output-dir', default='../outputs/glass_detection_segmentation')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GLASS DETECTION TRAINING")
    print("DFINE Segmentation Model (80 COCO + Segmentation -> 81 classes + Segmentation)")
    print("=" * 80)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Glass data: {args.glass_data}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"COCO retention: {args.coco_ratio:.0%}")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load segmentation model with criterion
    model, criterion = load_segmentation_model_from_checkpoint(
        args.config, args.checkpoint, device
    )
    
    # Expand to 81 classes
    model = expand_to_81_classes(model)
    
    # Update criterion num_classes
    if hasattr(criterion, 'num_classes'):
        criterion.num_classes = 81
    
    # Freeze strategically
    trainable_params = freeze_for_glass_training(
        model,
        unfreeze_last_n_decoder_layers=args.unfreeze_decoder_layers
    )
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Create datasets
    print(f"\n📊 Loading datasets...")
    
    # Load glass dataset
    if args.overfit:
        glass_train = GlassWallDataset(args.glass_data, split='all', class_id=80,
                                      augment=args.augment,
                                      augment_multiplier=args.augment_multiplier)
        glass_val = GlassWallDataset(args.glass_data, split='all', class_id=80, 
                                    augment=False)
    else:
        glass_train = GlassWallDataset(args.glass_data, split='train', class_id=80,
                                      augment=args.augment,
                                      augment_multiplier=args.augment_multiplier)
        glass_val = GlassWallDataset(args.glass_data, split='val', class_id=80, 
                                    augment=False)
    
    print(f"   ✅ Glass dataset loaded: {len(glass_train)} train, {len(glass_val)} val")
    split_label = 'all' if args.overfit else 'train'
    print(f"   🔁 Augment multiplier: {args.augment_multiplier} ({split_label} split)")
    
    # Test glass dataset
    try:
        test_img, test_target = glass_train[0]
        print(f"   ✅ Glass dataset test: image shape {test_img.shape}, {len(test_target['boxes'])} boxes")
    except Exception as e:
        print(f"   ⚠️  Glass dataset test failed: {e}")
        raise
    
    # Add COCO retention
    if os.path.exists(args.coco_data):
        print(f"   Loading COCO retention (ratio: {args.coco_ratio})...")
        try:
            coco_dataset = COCORetentionDataset(args.coco_data, max_images=args.coco_max_images)
            
            # Test COCO dataset
            try:
                test_coco_img, test_coco_target = coco_dataset[0]
                print(f"   ✅ COCO dataset test: image shape {test_coco_img.shape}, {len(test_coco_target['boxes'])} boxes")
            except Exception as e:
                print(f"   ⚠️  COCO dataset test failed: {e}")
                print(f"   Continuing without COCO retention...")
                coco_dataset = None
            
            if coco_dataset:
                coco_count = int(len(glass_train) * args.coco_ratio)
                coco_count = min(coco_count, len(coco_dataset))
                if coco_count > 0:
                    coco_subset = Subset(coco_dataset, range(coco_count))
                    train_dataset = ConcatDataset([glass_train, coco_subset])
                    print(
                        f"   ✅ Using mixed dataset: {len(glass_train)} glass + "
                        f"{coco_count} COCO (ratio {args.coco_ratio:.2f} of glass)"
                    )
                else:
                    train_dataset = glass_train
                    print("   ✅ Using glass-only dataset (COCO count is 0)")
            else:
                train_dataset = glass_train
        except Exception as e:
            print(f"   ⚠️  Failed to load COCO dataset: {e}")
            print(f"   Continuing with glass-only training...")
            train_dataset = glass_train
    else:
        print(f"   ⚠️  COCO data not found at {args.coco_data}")
        print(f"   Training on glass only (no COCO retention)")
        train_dataset = glass_train
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        glass_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(glass_val)}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Expected train batches: {(len(train_dataset) + args.batch_size - 1) // args.batch_size}")
    
    # Optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    warmup_epochs = min(5, args.epochs // 10)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\n📈 Training setup:")
    print(f"   Initial LR: {args.lr}")
    print(f"   Warmup epochs: {warmup_epochs}")
    print(f"   Trainable params: {trainable_params:,}")
    
    print(f"\n🚀 Starting training...")
    print("✅ Model: DFINE + Segmentation (frozen) + Glass Detection (new)")
    
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"📅 Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, avg_glass, loss_components = train_epoch(
            model, criterion, train_loader, optimizer, device, epoch
        )
        
        # Validate
        val_loss = validate_epoch(model, criterion, val_loader, device)
        
        scheduler.step()
        
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f} | Avg glass/batch: {avg_glass:.1f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss,
                          args.output_dir, 'best_glass_model.pth')
            print(f"   🏆 New best model! Val Loss: {best_loss:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, args.output_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
    
    print(f"\n{'='*80}")
    print(f"🎉 Training complete!")
    print(f"{'='*80}")
    print(f"🏆 Best validation loss: {best_loss:.4f}")
    print(f"📁 Models saved in: {args.output_dir}")
    print(f"\n💡 Your model now has:")
    print(f"   ✅ 80 COCO classes (preserved)")
    print(f"   ✅ 1 new class: glass_wall (class 80)")
    print(f"   ✅ Segmentation (Pascal Person Parts - preserved)")


if __name__ == '__main__':
    main()