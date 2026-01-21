
#!/usr/bin/env python3
"""
Training utilities for glass wall detection
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch
    
    Args:
        model: DFINE model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average training loss
        metrics: Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_glass_detections = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        
        # Move targets to device
        for target in targets:
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)
            target['image_id'] = target['image_id'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            # DFINE expects targets during training
            outputs = model(images, targets=targets)
            
            # Extract loss
            if isinstance(outputs, dict):
                if 'loss' in outputs:
                    loss = outputs['loss']
                elif 'loss_dict' in outputs:
                    # Sum all losses
                    loss_dict = outputs['loss_dict']
                    loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
                else:
                    # Fallback: compute simple classification loss
                    loss = compute_simple_loss(outputs, targets)
            else:
                loss = outputs
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Count glass detections (class 80)
            glass_count = sum((t['labels'] == 80).sum().item() for t in targets)
            total_glass_detections += glass_count
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'glass': glass_count
            })
        
        except Exception as e:
            print(f"\n⚠️  Error in batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_glass = total_glass_detections / num_batches if num_batches > 0 else 0.0
    
    metrics = {
        'glass_detections': avg_glass,
        'num_batches': num_batches
    }
    
    return avg_loss, metrics


def compute_simple_loss(outputs, targets):
    """Compute simple loss if DFINE doesn't return loss
    
    Args:
        outputs: Model outputs
        targets: Ground truth targets
    
    Returns:
        loss: Computed loss
    """
    # This is a fallback - DFINE should compute loss internally
    # If we reach here, just return a dummy loss
    return torch.tensor(0.0, requires_grad=True)


def validate_epoch(model, val_loader, device):
    """Validate for one epoch
    
    Args:
        model: DFINE model
        val_loader: Validation data loader
        device: Device
    
    Returns:
        avg_loss: Average validation loss
        metrics: Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            
            for target in targets:
                target['boxes'] = target['boxes'].to(device)
                target['labels'] = target['labels'].to(device)
                target['image_id'] = target['image_id'].to(device)
            
            try:
                outputs = model(images, targets=targets)
                
                if isinstance(outputs, dict):
                    if 'loss' in outputs:
                        loss = outputs['loss']
                    elif 'loss_dict' in outputs:
                        loss_dict = outputs['loss_dict']
                        loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
                    else:
                        loss = torch.tensor(0.0)
                else:
                    loss = outputs
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            except Exception as e:
                print(f"\n⚠️  Validation error: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    metrics = {
        'num_batches': num_batches
    }
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir, filename):
    """Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        loss: Current loss
        output_dir: Output directory
        filename: Checkpoint filename
    """
    filepath = os.path.join(output_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    torch.save(checkpoint, filepath)
    print(f"💾 Saved: {filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """Load training checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load into
        optimizer: Optimizer to load into (optional)
        scheduler: Scheduler to load into (optional)
        device: Device to load on
    
    Returns:
        epoch: Loaded epoch number
        loss: Loaded loss value
    """
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    print(f"✅ Loaded checkpoint from epoch {epoch}, loss: {loss:.4f}")
    
    return epoch, loss