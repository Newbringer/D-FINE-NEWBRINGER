#!/usr/bin/env python3
"""
PROPER Glass Wall Detection Training
Uses real DFINE model and loss - based on your working segmentation training
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add DFINE to path - same as your segmentation training
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    sys.path.insert(0, project_root)

# Import src module first to trigger all registrations
import src

# Import YAMLConfig properly (this triggers module registrations)
from src.core import YAMLConfig

# Import dataset
from dataset import GlassWallDataset, COCORetentionDataset, MixedDataset, collate_fn


def load_pretrained_dfine(config_path, checkpoint_path):
    """Load pretrained D-FINE model with criterion - copied from your segmentation training"""
    print(f"🚀 Loading pretrained D-FINE from {checkpoint_path}")
    
    cfg = YAMLConfig(config_path)
    model = cfg.model
    criterion = cfg.criterion  # Load DFINE's criterion for loss computation
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'ema' in checkpoint and 'module' in checkpoint['ema']:
        state_dict = checkpoint['ema']['module']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Check if checkpoint already has 81 classes (already expanded)
    needs_expansion = False
    for key in state_dict.keys():
        if 'dec_score_head.0.weight' in key:
            if state_dict[key].shape[0] == 81:
                print("   Checkpoint already has 81 classes - will expand model first")
                needs_expansion = True
            break
    
    # If checkpoint has 81 classes, expand model before loading
    if needs_expansion:
        model = expand_detection_head(model)
    
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded pretrained D-FINE model with criterion")
    return model, criterion


def expand_detection_head(model):
    """Expand detection head from 80 to 81 classes"""
    print("🔧 Expanding detection head: 80 -> 81 classes")
    
    expanded = False
    
    for name, module in model.named_modules():
        # Look for classification layers (including denoising_class_embed)
        if any(key in name.lower() for key in ['class_embed', 'cls', 'score']):
            # Check both Linear layers and Embedding layers
            if isinstance(module, nn.Linear) and module.out_features == 80:
                print(f"   Found: {name} (80 classes)")
                
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                # Create new layer with 81 classes
                in_features = module.in_features
                new_module = nn.Linear(in_features, 81)
                
                # Copy old weights (classes 0-79)
                with torch.no_grad():
                    new_module.weight[:80] = module.weight
                    if module.bias is not None:
                        new_module.bias[:80] = module.bias
                    
                    # Initialize new class (80) with small random values
                    nn.init.normal_(new_module.weight[80:], mean=0, std=0.01)
                    if new_module.bias is not None:
                        nn.init.zeros_(new_module.bias[80:])
                
                # Replace module
                setattr(parent, child_name, new_module)
                
                print(f"   ✅ Expanded to 81 classes")
                expanded = True
            
            # Also check for Embedding layers (used in denoising_class_embed)
            # denoising_class_embed has num_embeddings = num_classes + 1 (81 for 80 classes)
            elif isinstance(module, nn.Embedding) and module.num_embeddings == 81:
                print(f"   Found: {name} (81 embeddings, expanding to 82 for 81 classes)")
                
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                # Get padding_idx if it exists
                padding_idx = module.padding_idx if hasattr(module, 'padding_idx') else None
                
                # Create new embedding with 82 embeddings (num_classes + 1 = 81 + 1)
                embedding_dim = module.embedding_dim
                new_padding_idx = 81 if padding_idx is not None else None
                new_module = nn.Embedding(82, embedding_dim, padding_idx=new_padding_idx)
                
                # Copy old weights (embeddings 0-80)
                with torch.no_grad():
                    new_module.weight[:81] = module.weight
                    # Initialize new embedding (index 81) with small random values
                    if new_padding_idx != 81:  # Don't initialize padding index
                        nn.init.normal_(new_module.weight[81:82], mean=0, std=0.01)
                
                # Replace module
                setattr(parent, child_name, new_module)
                
                print(f"   ✅ Expanded to 82 embeddings (padding_idx={new_padding_idx})")
                expanded = True
    
    # Update num_classes attribute if it exists
    if hasattr(model, 'num_classes'):
        model.num_classes = 81
        print(f"   Updated model.num_classes = 81")
    
    # Check decoder if it exists
    if hasattr(model, 'decoder'):
        if hasattr(model.decoder, 'num_classes'):
            model.decoder.num_classes = 81
            print(f"   Updated decoder.num_classes = 81")
    
    if not expanded:
        print("   ⚠️  Warning: No classification layer found with 80 classes")
    
    return model


def freeze_model_except_classifiers(model, unfreeze_decoder=True, unfreeze_encoder_layers=2):
    """Freeze backbone, optionally unfreeze decoder and encoder layers
    
    Args:
        model: DFINE model
        unfreeze_decoder: If True, unfreeze entire decoder (RECOMMENDED for new classes like glass)
        unfreeze_encoder_layers: Number of encoder layers to unfreeze from end (0-6, default 2)
    """
    
    total_params = 0
    frozen_params = 0
    trainable_params = 0
    trainable_names = []
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        trainable = False
        
        # Always train classification/score heads
        if any(key in name.lower() for key in ['class_embed', 'cls', 'score']):
            trainable = True
        
        # Unfreeze entire decoder if requested (IMPORTANT for glass detection!)
        elif unfreeze_decoder and 'decoder' in name.lower():
            trainable = True
        
        # Unfreeze last N encoder layers if requested
        elif unfreeze_encoder_layers > 0 and 'encoder' in name.lower():
            # Try to find layer index in name
            import re
            layer_match = re.search(r'layers?[._](\d+)', name)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                if layer_idx >= (6 - unfreeze_encoder_layers):
                    trainable = True
        
        if trainable:
            # Only float/complex tensors can require gradients
            if param.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                param.requires_grad = True
                trainable_params += param.numel()
                trainable_names.append(name)
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"\n🔒 Freeze summary:")
    print(f"   Total params: {total_params:,}")
    print(f"   Frozen params: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"   Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"   Decoder unfrozen: {'YES ✅' if unfreeze_decoder else 'NO'}")
    print(f"   Encoder layers unfrozen: {unfreeze_encoder_layers}")
    print(f"\n   ✅ Sample trainable layers:")
    for name in trainable_names[:10]:  # Show first 10
        print(f"      - {name}")
    if len(trainable_names) > 10:
        print(f"      ... and {len(trainable_names)-10} more")
    
    return trainable_params


def train_epoch(model, criterion, dataloader, optimizer, device, epoch):
    """Train one epoch - using DFINE's actual loss computation with criterion"""
    model.train()
    criterion.train()
    
    total_loss = 0
    total_glass_count = 0
    num_batches = len(dataloader)
    
    # Track loss components for analysis
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
            # Forward pass - DFINE returns predictions
            outputs = model(images, targets=targets)
            
            # Compute loss using DFINE's criterion (already weighted inside)
            loss_dict = criterion(outputs, targets)
            
            # Sum all weighted loss components
            loss = sum(loss_dict.values())
            
            # Check if loss is valid
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
            
            # Track loss components for debugging
            for k, v in loss_dict.items():
                if k not in loss_components:
                    loss_components[k] = []
                loss_components[k].append(v.item())
            
            # Count glass detections in this batch
            glass_count = sum((t['labels'] == 80).sum().item() for t in targets)
            total_glass_count += glass_count
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'glass': glass_count
            })
        
        except Exception as e:
            print(f"\n⚠️  Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_glass = total_glass_count / num_batches if num_batches > 0 else 0.0
    
    # Print loss component breakdown for analysis
    if loss_components and epoch % 5 == 0:  # Every 5 epochs
        print(f"\n   📊 Loss component breakdown (epoch {epoch+1}):")
        for k, values in loss_components.items():
            avg_val = np.mean(values)
            print(f"      {k}: {avg_val:.4f}")
    
    return avg_loss, avg_glass, loss_components


def validate_epoch(model, criterion, dataloader, device):
    """Validate one epoch
    
    Note: DFINE needs to stay in training mode even during validation
    because it requires targets to compute loss. We use torch.no_grad()
    to disable gradient computation while keeping the forward pass behavior.
    """
    # Keep model in training mode but disable gradients
    # DFINE's eval() mode doesn't accept targets, but we need them for loss
    model.train()
    criterion.train()
    
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
                # Forward pass - model must be in training mode to accept targets
                outputs = model(images, targets=targets)
                
                # Compute loss using DFINE's criterion (already weighted inside)
                loss_dict = criterion(outputs, targets)
                
                # Sum all weighted loss components
                loss = sum(loss_dict.values())
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            except Exception as e:
                print(f"\n⚠️  Validation error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir, filename='best_model.pth'):
    """Save checkpoint"""
    filepath = os.path.join(output_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'num_classes': 81,
    }, filepath)
    
    print(f"💾 Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='PROPER Glass Wall Detection Training')
    
    # Model - using local glass wall config
    parser.add_argument('--config', default='../models/dfine_hgnetv2_x_obj2coco.yml',
                       help='DFINE config file')
    parser.add_argument('--checkpoint', default='../models/dfine_0.73.pth',
                       help='Path to checkpoint with segmentation')
    
    # Data
    parser.add_argument('--glass-data', default='../data/glass_wall',
                       help='Glass wall dataset')
    parser.add_argument('--use-coco-retention', action='store_true',
                       help='Mix with COCO data')
    parser.add_argument('--coco-data', default='../data/coco',
                       help='COCO dataset path')
    parser.add_argument('--coco-ratio', type=float, default=0.3,
                       help='Ratio of COCO samples')
    parser.add_argument('--coco-max-images', type=int, default=None,
                        help='Override max COCO images (default: auto from ratio)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5, 
                        help='Learning rate (default: 1e-5 for stability)')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Enable light training augmentation (default: False)')
    parser.add_argument('--overfit', action='store_true', default=False,
                        help='Train/validate on full dataset for overfit check')
    parser.add_argument('--disable-denoising', action='store_true', default=False,
                        help='Disable denoising queries for small dataset overfit')
    parser.add_argument('--unfreeze-decoder', action='store_true', default=False,
                        help='Unfreeze decoder (default: False for stability)')
    parser.add_argument('--unfreeze-encoder-layers', type=int, default=0,
                        help='Number of encoder layers to unfreeze (0-6, default 0)')
    
    # Output
    parser.add_argument('--output-dir', default='../outputs/glass_detection_proper')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PROPER GLASS WALL DETECTION TRAINING")
    print("Using REAL DFINE model and loss")
    print("=" * 80)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Glass data: {args.glass_data}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"COCO retention: {args.use_coco_retention}")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load DFINE model and criterion properly (like your segmentation training)
    print("\n📦 Loading DFINE model...")
    model, criterion = load_pretrained_dfine(args.config, args.checkpoint)
    
    # Expand detection head if not already expanded
    # Check if already 81 classes
    already_expanded = False
    for name, module in model.named_modules():
        if 'dec_score_head.0' in name and isinstance(module, nn.Linear):
            if module.out_features == 81:
                already_expanded = True
                print("   Model already has 81 classes - skipping expansion")
            break
    
    if not already_expanded:
        model = expand_detection_head(model)
    
    # Update criterion num_classes to match expanded model
    if hasattr(criterion, 'num_classes'):
        criterion.num_classes = 81
        print(f"   Updated criterion.num_classes = 81")

    # Optionally disable denoising queries (helps small datasets overfit)
    if args.disable_denoising and hasattr(model, 'decoder'):
        model.decoder.num_denoising = 0
        model.decoder.label_noise_ratio = 0.0
        model.decoder.box_noise_scale = 0.0
        print("   Disabled denoising queries for decoder")
    
    # Freeze everything except classifiers
    trainable_params = freeze_model_except_classifiers(
        model, 
        unfreeze_decoder=args.unfreeze_decoder,
        unfreeze_encoder_layers=args.unfreeze_encoder_layers
    )
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Create datasets
    print(f"\n📊 Loading datasets...")
    if args.overfit:
        glass_train = GlassWallDataset(args.glass_data, split='all', class_id=80, augment=args.augment)
        glass_val = GlassWallDataset(args.glass_data, split='all', class_id=80, augment=False)
    else:
        glass_train = GlassWallDataset(args.glass_data, split='train', class_id=80, augment=args.augment)
        glass_val = GlassWallDataset(args.glass_data, split='val', class_id=80, augment=False)
    
    if args.use_coco_retention and os.path.exists(args.coco_data):
        print(f"   Using COCO retention (ratio: {args.coco_ratio})")
        # Auto-scale COCO size to match desired ratio
        if args.coco_ratio >= 1.0 or args.coco_ratio <= 0.0:
            raise ValueError("--coco-ratio must be between 0 and 1 (exclusive)")
        target_coco = int(len(glass_train) * args.coco_ratio / (1 - args.coco_ratio))
        coco_max_images = args.coco_max_images if args.coco_max_images is not None else max(1, target_coco)
        print(f"   COCO max images: {coco_max_images} (target ratio)")
        coco_dataset = COCORetentionDataset(args.coco_data, max_images=coco_max_images)
        train_dataset = MixedDataset(glass_train, coco_dataset, 
                                    glass_ratio=1-args.coco_ratio)
    else:
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
    
    # Optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Warmup + Cosine Annealing scheduler
    warmup_epochs = min(5, args.epochs // 10)  # 5 epochs or 10% of total
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"📈 Learning rate schedule:")
    print(f"   Initial LR: {args.lr}")
    print(f"   Warmup epochs: {warmup_epochs}")
    print(f"   Min LR: {args.lr * 1e-2:.2e}")
    
    print(f"\n🚀 Starting PROPER training...")
    print("✅ Using REAL DFINE loss computation")
    
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, avg_glass, loss_components = train_epoch(model, criterion, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss = validate_epoch(model, criterion, val_loader, device)
        
        scheduler.step()
        
        print(f"📊 Train Loss: {train_loss:.4f} | Avg glass/batch: {avg_glass:.1f}")
        print(f"📊 Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, 
                          args.output_dir, 'best_model.pth')
            print(f"🏆 New best model! Val Loss: {best_loss:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, args.output_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
    
    print(f"\n🎉 Training complete!")
    print(f"🏆 Best validation loss: {best_loss:.4f}")
    print(f"📁 Models saved in: {args.output_dir}")


if __name__ == '__main__':
    main()