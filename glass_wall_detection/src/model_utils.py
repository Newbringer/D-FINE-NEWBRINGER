#!/usr/bin/env python3
"""
Model utilities for glass wall detection
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add DFINE to path (D-FINE-NEWBRINGER/src/)
sys.path.insert(0, '../../src')
from core import YAMLConfig


def load_model_with_segmentation(checkpoint_path, config_path, device):
    """Load DFINE model with segmentation head
    
    Args:
        checkpoint_path: Path to checkpoint (dfine_0.73.pth)
        config_path: Path to DFINE config
        device: Device to load model on
    
    Returns:
        model: Loaded model with segmentation head
    """
    print(f"   Loading config: {config_path}")
    cfg = YAMLConfig(config_path)
    model = cfg.model
    
    print(f"   Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'ema' in checkpoint and 'module' in checkpoint['ema']:
        state_dict = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load state dict (strict=False to handle segmentation head)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"   ⚠️  Missing keys: {len(missing)}")
        # This is expected if segmentation head has different keys
    if unexpected:
        print(f"   ⚠️  Unexpected keys: {len(unexpected)}")
    
    model = model.to(device)
    model.train()
    
    print(f"   ✅ Model loaded successfully")
    
    # Print model structure info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    return model


def expand_detection_head(model):
    """Expand detection head from 80 to 81 classes
    
    Args:
        model: DFINE model
    """
    print("   Searching for classification layers...")
    
    expanded = False
    
    # DFINE typically has class_embed in decoder
    for name, module in model.named_modules():
        # Look for classification layers
        if any(key in name.lower() for key in ['class_embed', 'cls', 'score']):
            # Check if it's a Linear layer with 80 output features
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
    
    # Also check if model has num_classes attribute
    if hasattr(model, 'num_classes'):
        model.num_classes = 81
        print(f"   Updated model.num_classes = 81")
    
    if not expanded:
        print("   ⚠️  Warning: No classification layer found with 80 classes")
        print("   This might be okay if your model already has 81 classes")
    
    return model


def freeze_model(model):
    """Freeze everything except detection head classifier
    
    Args:
        model: DFINE model
    
    Returns:
        trainable_params: Number of trainable parameters
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # Only train detection head classification layers
        if any(key in name.lower() for key in ['class_embed', 'cls', 'score']):
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"   ✅ Trainable: {name}")
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"\n   Freeze summary:")
    print(f"   Total params: {total_params:,}")
    print(f"   Frozen params: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"   Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    return trainable_params


def get_model_info(model):
    """Get model information
    
    Args:
        model: DFINE model
    
    Returns:
        info: Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }
    
    return info


def print_model_structure(model, max_depth=2):
    """Print model structure
    
    Args:
        model: DFINE model
        max_depth: Maximum depth to print
    """
    print("\n" + "=" * 80)
    print("MODEL STRUCTURE")
    print("=" * 80)
    
    def print_module(module, name='', depth=0, max_depth=2):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        
        # Get module type
        module_type = type(module).__name__
        
        # Count parameters
        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        
        if params > 0:
            status = "🔓" if trainable > 0 else "🔒"
            print(f"{indent}{status} {name} ({module_type}) - {params:,} params")
        elif depth < max_depth:
            print(f"{indent}📁 {name} ({module_type})")
        
        # Recurse into children
        if depth < max_depth:
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                print_module(child_module, full_name, depth + 1, max_depth)
    
    print_module(model, max_depth=max_depth)
    print("=" * 80)