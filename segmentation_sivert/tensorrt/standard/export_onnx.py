#!/usr/bin/env python3
"""
DFINE Segmentation Model ONNX Exporter - FIXED VERSION

Converts a DFINE + Segmentation PyTorch model to ONNX format for TensorRT conversion.
Now properly loads hyperparameters from checkpoint to match training configuration.

Usage: 
    python export_onnx.py -r outputs/dfine_segmentation_optuna/best_model.pth --original-weights models/dfine_x_obj2coco.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path
sys.path.append('src')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DFINE Segmentation Model ONNX Exporter')
    parser.add_argument('-c', '--config', type=str, default="configs/dfine_hgnetv2_x_obj2coco.yml",
                        help='Path to original DFINE configuration YAML')
    parser.add_argument('-r', '--resume', type=str, required=True,
                        help='Path to trained segmentation model weights (.pth)')
    parser.add_argument('--original-weights', default="models/dfine_x_obj2coco.pth", type=str,
                        help='Path to original DFINE model weights (.pth)')
    parser.add_argument('-o', '--output', type=str, default="model.onnx",
                        help='Path to save the ONNX model (default: auto-generated from resume path)')
    parser.add_argument('--check', action='store_true', default=True,
                        help='Check the exported ONNX model')
    parser.add_argument('--simplify', action='store_true', default=True,
                        help='Simplify the ONNX model')
    parser.add_argument('--input-shape', type=str, default='1,3,640,640',
                        help='Input shape for the model (default: 1,3,640,640)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--num-classes', type=int, default=7,
                        help='Number of segmentation classes (default: 7 for Pascal Person Parts)')
    return parser.parse_args()

# Import segmentation components from your training script
class SimplifiedASPP(nn.Module):
    """Simplified ASPP for better speed/accuracy balance"""
    
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12]):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for dilation in dilations:
            if dilation == 1:
                conv = nn.Conv2d(in_channels, out_channels//len(dilations), 1, bias=False)
            else:
                conv = nn.Conv2d(in_channels, out_channels//len(dilations), 3, 
                               padding=dilation, dilation=dilation, bias=False)
            self.convs.append(nn.Sequential(
                conv, 
                nn.BatchNorm2d(out_channels//len(dilations)), 
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels//len(dilations), 1, bias=False),
            nn.BatchNorm2d(out_channels//len(dilations)),
            nn.ReLU(inplace=True)
        )
        
        # Final projection
        total_channels = out_channels//len(dilations) * (len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x):
        h, w = x.shape[-2:]
        features = []
        
        # Apply dilated convolutions
        for conv in self.convs:
            features.append(conv(x))
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        features.append(global_feat)
        
        # Combine and project
        combined = torch.cat(features, dim=1)
        return self.project(combined)


class ImprovedFPN(nn.Module):
    """Improved FPN with better feature fusion"""
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False) 
            for in_ch in in_channels_list
        ])
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in in_channels_list
        ])
        
        # Feature fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(in_channels_list)))
        
    def forward(self, features):
        # Build laterals
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Build top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], 
                mode='bilinear', align_corners=False
            )
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Apply final convs
        outputs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        
        # Weighted feature fusion
        target_size = outputs[0].shape[-2:]
        fused_features = []
        
        for i, feat in enumerate(outputs):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            fused_features.append(feat * self.fusion_weights[i])
        
        # Return highest resolution feature with weighted fusion
        return sum(fused_features)


class EnhancedSegmentationHead(nn.Module):
    """Enhanced but stable segmentation head"""
    
    def __init__(self, in_channels_list, num_classes=7, feature_dim=256, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        
        # Improved FPN
        self.fpn = ImprovedFPN(in_channels_list, feature_dim)
        
        # Simplified ASPP for multi-scale context
        self.aspp = SimplifiedASPP(feature_dim, feature_dim)
        
        # Enhanced decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_dim, feature_dim//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim//2),
            nn.ReLU(inplace=True),
            
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(feature_dim//2, num_classes, 1)
        )
    
    def forward(self, features):
        # Multi-scale feature fusion
        fused_feature = self.fpn(features)
        
        # Apply simplified ASPP
        context_feature = self.aspp(fused_feature)
        
        # Generate segmentation
        seg_logits = self.decoder(context_feature)
        
        return seg_logits


class DFineWithSegmentation(nn.Module):
    """D-FINE with enhanced segmentation"""
    
    def __init__(self, dfine_model, seg_head, freeze_detection=True):
        super().__init__()
        self.dfine_model = dfine_model
        self.seg_head = seg_head
        self.freeze_detection = freeze_detection
    
    def forward(self, x, targets=None):
        # Get backbone features
        backbone_features = self.dfine_model.backbone(x)
        
        outputs = {}
        
        # Detection branch
        det_outputs = self.dfine_model(x)
        outputs.update(det_outputs)
        
        # Segmentation branch
        seg_logits = self.seg_head(backbone_features)
        
        # Upsample to input resolution
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        outputs['segmentation'] = seg_logits
        
        return outputs


def get_actual_backbone_channels(model):
    """Get actual backbone output channels"""
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        backbone_features = model.backbone(dummy_input)
    
    actual_channels = [feat.shape[1] for feat in backbone_features]
    return actual_channels


def load_pretrained_dfine(config_path, checkpoint_path):
    """Load pretrained D-FINE model"""
    print(f"🚀 Loading pretrained D-FINE from {checkpoint_path}")
    
    from src.core import YAMLConfig
    cfg = YAMLConfig(config_path)
    model = cfg.model
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'ema' in checkpoint and 'module' in checkpoint['ema']:
        state_dict = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded pretrained D-FINE model")
    return model


def load_hyperparameters_from_checkpoint(checkpoint_path):
    """Load hyperparameters from saved checkpoint"""
    print(f"📚 Loading hyperparameters from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract hyperparameters
    if 'hyperparameters' in checkpoint:
        hyperparams = checkpoint['hyperparameters']
        print(f"✅ Found saved hyperparameters:")
        for key, value in hyperparams.items():
            print(f"   {key}: {value}")
        return hyperparams
    else:
        print("⚠️  No hyperparameters found in checkpoint, using defaults")
        return {
            'feature_dim': 256,
            'dropout_rate': 0.1
        }


def create_segmentation_model(config_path, original_checkpoint_path, seg_checkpoint_path, num_classes=7):
    """Create segmentation model and load trained weights with correct hyperparameters"""
    
    # STEP 1: Load hyperparameters from the segmentation checkpoint
    hyperparams = load_hyperparameters_from_checkpoint(seg_checkpoint_path)
    
    # Extract the specific hyperparameters we need
    feature_dim = hyperparams.get('feature_dim', 256)
    dropout_rate = hyperparams.get('dropout_rate', 0.1)
    
    print(f"🏗️  Creating model with feature_dim={feature_dim}, dropout_rate={dropout_rate}")
    
    # STEP 2: Load original DFINE model
    det_model = load_pretrained_dfine(config_path, original_checkpoint_path)
    backbone_channels = get_actual_backbone_channels(det_model)
    
    print(f"🏗️  Using backbone channels: {backbone_channels}")
    
    # STEP 3: Create segmentation head with CORRECT hyperparameters
    seg_head = EnhancedSegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=num_classes,
        feature_dim=feature_dim,  # Use the saved hyperparameter!
        dropout_rate=dropout_rate  # Use the saved hyperparameter!
    )
    
    # STEP 4: Create combined model
    model = DFineWithSegmentation(
        dfine_model=det_model,
        seg_head=seg_head,
        freeze_detection=False  # Don't freeze during inference
    )
    
    # STEP 5: Load trained segmentation weights
    print(f"📚 Loading trained segmentation weights from {seg_checkpoint_path}")
    checkpoint = torch.load(seg_checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Now the model architecture should match the checkpoint!
    model.load_state_dict(state_dict, strict=True)
    print(f"✅ Loaded trained segmentation model successfully!")
    
    return model, hyperparams


def export_onnx(args):
    """Export the segmentation model to ONNX format"""
    
    # Create the segmentation model with correct hyperparameters
    model, hyperparams = create_segmentation_model(
        args.config, 
        args.original_weights, 
        args.resume, 
        args.num_classes
    )
    
    # Create a wrapper model for ONNX export that includes postprocessing
    class SegmentationModelWrapper(nn.Module):
        def __init__(self, seg_model):
            super().__init__()
            self.seg_model = seg_model.eval()
            
            # Get the original postprocessor from DFINE config
            from src.core import YAMLConfig
            cfg = YAMLConfig(args.config)
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            # Run the segmentation model
            outputs = self.seg_model(images)
            
            # Extract detection outputs for postprocessing
            det_outputs = {k: v for k, v in outputs.items() if k != 'segmentation'}
            
            # Process detection outputs
            processed_det = self.postprocessor(det_outputs, orig_target_sizes)
            
            # Get segmentation outputs
            seg_logits = outputs['segmentation']
            
            # Apply softmax to get probabilities
            seg_probs = torch.softmax(seg_logits, dim=1)
            
            # Get segmentation predictions
            seg_preds = torch.argmax(seg_logits, dim=1)
            
            return processed_det[0], processed_det[1], processed_det[2], seg_probs, seg_preds

    wrapper_model = SegmentationModelWrapper(model)
    wrapper_model.eval()
    
    # Parse input shape
    input_shape = [int(x) for x in args.input_shape.split(",")]
    print(f"Using input shape: {input_shape}")
    
    # Create dummy input
    data = torch.rand(*input_shape)
    size = torch.tensor([[input_shape[3], input_shape[2]]])  # width, height format
    
    # Run a test forward pass
    print("Running test forward pass...")
    with torch.no_grad():
        test_outputs = wrapper_model(data, size)
        print(f"Test outputs: {len(test_outputs)} tensors")
        for i, out in enumerate(test_outputs):
            print(f"  Output {i}: shape {out.shape}, dtype {out.dtype}")
    
    # Define dynamic axes for ONNX export
    dynamic_axes = {
        "images": {0: "N"},  # Batch dimension
        "orig_target_sizes": {0: "N"},  # Batch dimension
        "labels": {0: "N"},
        "boxes": {0: "N"},
        "scores": {0: "N"},
        "seg_probs": {0: "N"},
        "seg_preds": {0: "N"},
    }
    
    # Determine output file name
    if args.output:
        output_file = args.output
    else:
        # Create a more descriptive filename with hyperparameters
        base_name = args.resume.replace(".pth", "")
        feature_dim = hyperparams.get('feature_dim', 256)
        output_file = f"{base_name}_segmentation_fd{feature_dim}.onnx"
    
    print(f"Exporting segmentation model to ONNX: {output_file}")
    print(f"Model hyperparameters: feature_dim={hyperparams.get('feature_dim', 256)}, dropout_rate={hyperparams.get('dropout_rate', 0.1)}")
    
    torch.onnx.export(
        wrapper_model,
        (data, size),
        output_file,
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores", "seg_probs", "seg_preds"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
    )
    print("ONNX export complete!")
    
    # Check the ONNX model if requested
    if args.check:
        try:
            import onnx
            print("Checking ONNX model...")
            onnx_model = onnx.load(output_file)
            onnx.checker.check_model(onnx_model)
            print("ONNX model check successful!")
        except ImportError:
            print("Warning: onnx package not found, skipping model check.")
            print("Install onnx with: pip install onnx")
        except Exception as e:
            print(f"Error checking ONNX model: {e}")
    
    # Simplify the ONNX model if requested
    if args.simplify:
        try:
            import onnx
            import onnxsim
            print("Simplifying ONNX model...")
            
            # Create input shapes dictionary for simplification
            input_shapes = {
                "images": input_shape, 
                "orig_target_sizes": [input_shape[0], 2]
            }
            
            # Simplify the model
            onnx_model = onnx.load(output_file)
            model_simp, check = onnxsim.simplify(
                onnx_model, 
                test_input_shapes=input_shapes
            )
            
            if check:
                onnx.save(model_simp, output_file)
                print(f"Simplified ONNX model saved to: {output_file}")
            else:
                print("Warning: Simplified ONNX model could not be validated.")
        except ImportError:
            print("Warning: onnx-simplifier package not found, skipping model simplification.")
            print("Install onnx-simplifier with: pip install onnx-simplifier")
        except Exception as e:
            print(f"Error simplifying ONNX model: {e}")
    
    print("Done!")
    return output_file

if __name__ == "__main__":
    args = parse_args()
    export_onnx(args)