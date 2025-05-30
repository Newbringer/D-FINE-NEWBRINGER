"""
Train D-FINE with Body Parts Segmentation - CHANNEL FIXED VERSION
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.core import YAMLConfig


class PascalPersonPartsDataset(torch.utils.data.Dataset):
    """Pascal Person Parts Dataset Loader"""
    
    def __init__(self, root_dir, split='train', image_size=640):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # Get image paths
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.mask_dir = os.path.join(root_dir, 'masks', split)
        
        self.image_names = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
        print(f"📊 Loaded {len(self.image_names)} {split} samples")
        
        # Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_name = img_name.replace('.jpg', '.png')
        
        # Load image and mask
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image
        import cv2
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        print(f"🏗️  FPN expects input channels: {in_channels_list}")
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) 
            for in_ch in in_channels_list
        ])
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
        
    def forward(self, features):
        """
        Args:
            features: List of features from HGNetv2
        """
        print(f"🔍 FPN received {len(features)} features:")
        for i, feat in enumerate(features):
            print(f"   Feature {i}: {feat.shape}")
        
        # Build laterals
        laterals = []
        for i, (conv, feat) in enumerate(zip(self.lateral_convs, features)):
            print(f"   Processing feature {i}: {feat.shape} -> conv expecting {conv.in_channels} channels")
            lateral = conv(feat)
            laterals.append(lateral)
        
        # Build top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher level feature
            upsampled = F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], 
                mode='bilinear', align_corners=False
            )
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Apply final convs
        outputs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        
        return outputs


class BodyPartsSegmentationHead(nn.Module):
    """
    Body Parts Segmentation Head for D-FINE
    6 body parts: head, torso, arms, hands, legs, feet
    """
    
    def __init__(
        self, 
        in_channels_list,  # HGNetv2 output channels 
        num_classes=7,     # 6 body parts + background
        feature_dim=256,
        dropout_rate=0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels_list = in_channels_list
        
        print(f"🏗️  Creating segmentation head with channels: {in_channels_list}")
        
        # Feature Pyramid Network
        self.fpn = FPN(in_channels_list, feature_dim)
        
        # Segmentation decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim//2, 3, padding=1),
            nn.BatchNorm2d(feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(feature_dim//2, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from HGNetv2 backbone
        Returns:
            Segmentation logits
        """
        # Use FPN to fuse multi-scale features
        fpn_features = self.fpn(features)
        
        # Use the highest resolution feature (P2) for final prediction
        final_feature = fpn_features[0]
        
        # Generate segmentation prediction
        seg_logits = self.decoder(final_feature)
        
        return seg_logits


class DFineWithSegmentation(nn.Module):
    """
    D-FINE with added body parts segmentation head
    """
    
    def __init__(self, dfine_model, seg_head, freeze_detection=True):
        super().__init__()
        self.dfine_model = dfine_model
        self.seg_head = seg_head
        self.freeze_detection = freeze_detection
        
        # Freeze detection components if needed
        if freeze_detection:
            self._freeze_detection()
    
    def _freeze_detection(self):
        """Freeze all detection-related parameters"""
        frozen_params = 0
        total_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if 'seg_head' not in name:
                param.requires_grad = False
                frozen_params += param.numel()
        
        trainable_params = total_params - frozen_params
        
        print(f"🔒 Frozen detection components:")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   ❄️  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"   🎯 Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    def forward(self, x, targets=None):
        # Get backbone features from D-FINE
        backbone_features = self.dfine_model.backbone(x)
        
        outputs = {}
        
        # Detection branch (frozen during training)
        if not self.training or not self.freeze_detection:
            # Get detection outputs from the full D-FINE model
            with torch.no_grad() if self.freeze_detection else torch.enable_grad():
                det_outputs = self.dfine_model(x)
                outputs.update(det_outputs)  # pred_logits, pred_boxes, etc.
        
        # Segmentation branch (trainable)
        seg_logits = self.seg_head(backbone_features)
        
        # Upsample to input resolution
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        outputs['segmentation'] = seg_logits
        
        return outputs


class SegmentationLoss(nn.Module):
    """Segmentation loss with class weighting"""
    
    def __init__(self, ignore_index=255, weight=None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
    
    def forward(self, pred, target):
        return self.criterion(pred, target)


def compute_miou(pred, target, num_classes=7):
    """Compute mean IoU"""
    pred = torch.argmax(pred, dim=1)
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            ious.append(intersection / union)
    
    return torch.tensor(ious).mean() if ious else torch.tensor(0.0)


def get_actual_backbone_channels(model):
    """Get actual backbone output channels by running a forward pass"""
    print("🔍 Detecting actual backbone channels...")
    
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        backbone_features = model.backbone(dummy_input)
    
    actual_channels = [feat.shape[1] for feat in backbone_features]
    print(f"✅ Detected actual backbone channels: {actual_channels}")
    
    return actual_channels


def load_pretrained_dfine(config_path, checkpoint_path):
    """Load pretrained D-FINE model"""
    print(f"🚀 Loading pretrained D-FINE from {checkpoint_path}")
    
    # Load config
    cfg = YAMLConfig(config_path)
    
    # Create model
    model = cfg.model
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'ema' in checkpoint and 'module' in checkpoint['ema']:
        state_dict = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
    
    print(f"✅ Loaded pretrained D-FINE model")
    return model


def inspect_model_structure(model):
    """Inspect the D-FINE model structure"""
    print("🔍 D-FINE Model Structure:")
    for name, module in model.named_children():
        print(f"   - {name}: {type(module).__name__}")
    
    # Get actual backbone channels by running forward pass
    actual_channels = get_actual_backbone_channels(model)
    
    return actual_channels


def create_segmentation_model(config_path, checkpoint_path, num_classes=7):
    """Create D-FINE model with segmentation head"""
    
    # Load pretrained detection model
    det_model = load_pretrained_dfine(config_path, checkpoint_path)
    
    # Get actual backbone channels by running forward pass
    backbone_channels = inspect_model_structure(det_model)
    
    print(f"🏗️  Using actual backbone channels: {backbone_channels}")
    
    # Create segmentation head with correct channels
    seg_head = BodyPartsSegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=num_classes,
        feature_dim=256
    )
    
    # Create combined model
    model = DFineWithSegmentation(
        dfine_model=det_model,
        seg_head=seg_head,
        freeze_detection=True
    )
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_miou = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        seg_pred = outputs['segmentation']
        
        # Calculate loss
        loss = criterion(seg_pred, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            miou = compute_miou(seg_pred, masks)
        
        total_loss += loss.item()
        total_miou += miou.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'mIoU': f'{miou.item():.3f}'
        })
        
        # Log to wandb
        if batch_idx % 20 == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_miou': miou.item(),
                'train/lr': optimizer.param_groups[0]['lr']
            })
        
        # Only print debug info for first batch
        if batch_idx == 0 and epoch == 0:
            print("🔍 Debug info disabled after first batch")
            # Remove debug prints from FPN
            model.seg_head.fpn.forward = lambda features: model.seg_head.fpn.__class__.forward(model.seg_head.fpn, features)
    
    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    
    return avg_loss, avg_miou


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0
    total_miou = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            seg_pred = outputs['segmentation']
            
            # Calculate loss and metrics
            loss = criterion(seg_pred, masks)
            miou = compute_miou(seg_pred, masks)
            
            total_loss += loss.item()
            total_miou += miou.item()
    
    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    
    return avg_loss, avg_miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/dfine_hgnetv2_x_obj2coco.yml', 
                       help='Path to D-FINE config file')
    parser.add_argument('--checkpoint', default='models/dfine_x_obj2coco.pth',
                       help='Path to pretrained D-FINE checkpoint')
    parser.add_argument('--dataset', default='datasets/pascal_person_parts',
                       help='Path to Pascal Person Parts dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', default='outputs/dfine_segmentation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project='dfine-body-parts-segmentation',
        config=vars(args),
        name=f'dfine_seg_{args.epochs}ep_channel_fixed'
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Create model
    print("🏗️  Creating segmentation model...")
    model = create_segmentation_model(
        args.config, 
        args.checkpoint, 
        num_classes=7
    )
    model = model.to(device)
    
    # Create datasets
    print("📊 Loading datasets...")
    train_dataset = PascalPersonPartsDataset(
        args.dataset, split='train', image_size=args.image_size
    )
    val_dataset = PascalPersonPartsDataset(
        args.dataset, split='val', image_size=args.image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create loss and optimizer
    criterion = SegmentationLoss()
    
    # Only optimize segmentation head parameters
    seg_params = [p for p in model.seg_head.parameters() if p.requires_grad]
    optimizer = optim.AdamW(seg_params, lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"🎯 Training {sum(p.numel() for p in seg_params):,} segmentation parameters")
    
    # Training loop
    best_miou = 0
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_miou = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_miou = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/miou': train_miou,
            'val/loss': val_loss,
            'val/miou': val_miou,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"📊 Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.3f}")
        print(f"📊 Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.3f}")
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'config': vars(args)
            }, f'{args.output_dir}/best_model.pth')
            print(f"💾 Saved best model with mIoU: {best_miou:.3f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'miou': val_miou,
            }, f'{args.output_dir}/checkpoint_epoch_{epoch+1}.pth')
    
    print(f"🎉 Training completed! Best mIoU: {best_miou:.3f}")
    wandb.finish()


if __name__ == '__main__':
    main()