#!/usr/bin/env python3
"""
Amodal Offset Prediction Head
Predicts full extent of occluded humans from visible boxes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torchvision.ops import roi_align


class AmodalOffsetHead(nn.Module):
    """
    Predicts amodal box offset from visible detections
    
    Architecture:
    - RoI features from DFINE backbone (frozen)
    - Visible box embedding
    - Predict offset to expand to full amodal extent
    """
    
    def __init__(self, 
                 in_channels: int = 256,
                 hidden_dim: int = 512,
                 roi_size: int = 7):
        super().__init__()
        
        self.roi_size = roi_size
        self.in_channels = in_channels
        
        # Process RoI features
        self.roi_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Encode visible box geometry
        self.box_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Fusion and offset prediction
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # Predict offset [delta_x1, delta_y1, delta_x2, delta_y2]
        # Negative = expand left/up, positive = expand right/down
        self.offset_head = nn.Linear(hidden_dim // 2, 4)
        
        # Predict occlusion score
        self.occlusion_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Predict confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, 
                roi_features: torch.Tensor,
                visible_boxes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            roi_features: [N, C, H, W] RoI features
            visible_boxes: [N, 4] Visible boxes [x1, y1, x2, y2] normalized
        
        Returns:
            Dict with amodal_boxes, offset, occlusion, confidence
        """
        batch_size = roi_features.size(0)
        
        # Process RoI features
        roi_feat = self.roi_conv(roi_features)  # [N, hidden_dim, 1, 1]
        roi_feat = roi_feat.flatten(1)  # [N, hidden_dim]
        
        # Encode visible box
        box_feat = self.box_encoder(visible_boxes)  # [N, hidden_dim//2]
        
        # Fuse features
        combined = torch.cat([roi_feat, box_feat], dim=1)
        features = self.fusion(combined)
        
        # Predict offset
        offset = self.offset_head(features)  # [N, 4]
        
        # Compute amodal boxes (avoid in-place ops for gradient computation)
        amodal_boxes = visible_boxes + offset
        
        # Ensure amodal >= visible (can only expand, not shrink)
        amodal_boxes = torch.stack([
            torch.min(amodal_boxes[:, 0], visible_boxes[:, 0]),  # x1
            torch.min(amodal_boxes[:, 1], visible_boxes[:, 1]),  # y1
            torch.max(amodal_boxes[:, 2], visible_boxes[:, 2]),  # x2
            torch.max(amodal_boxes[:, 3], visible_boxes[:, 3])   # y2
        ], dim=1)
        
        # Clamp to [0, 1]
        amodal_boxes = amodal_boxes.clamp(0, 1)
        
        # Predict occlusion and confidence
        occlusion = self.occlusion_head(features).squeeze(-1)
        confidence = self.confidence_head(features).squeeze(-1)
        
        return {
            'amodal_boxes': amodal_boxes,
            'visible_boxes': visible_boxes,
            'offset': offset,
            'occlusion_scores': occlusion,
            'confidence_scores': confidence
        }


class AmodalLoss(nn.Module):
    """Loss function for amodal prediction"""
    
    def __init__(self,
                 weight_offset: float = 10.0,
                 weight_giou: float = 5.0,
                 weight_occlusion: float = 2.0):
        super().__init__()
        self.weight_offset = weight_offset
        self.weight_giou = weight_giou
        self.weight_occlusion = weight_occlusion
    
    def box_iou(self, boxes1, boxes2):
        """Compute IoU between boxes [x1, y1, x2, y2]"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        union = area1 + area2 - inter
        iou = inter / (union + 1e-6)
        
        return iou
    
    def generalized_box_iou(self, boxes1, boxes2):
        """Compute GIoU"""
        iou = self.box_iou(boxes1, boxes2)
        
        # Enclosing box
        lt = torch.min(boxes1[:, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        area_c = wh[:, 0] * wh[:, 1]
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - iou * (area1 + area2) / (1 + iou + 1e-6)
        
        giou = iou - (area_c - union) / (area_c + 1e-6)
        
        return giou
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with amodal_boxes, offset, occlusion_scores
            targets: dict with amodal_boxes, occlusion_scores
        """
        pred_amodal = predictions['amodal_boxes']
        pred_offset = predictions['offset']
        pred_occlusion = predictions['occlusion_scores']
        
        target_amodal = targets['amodal_boxes']
        target_occlusion = targets['occlusion_scores']
        visible_boxes = predictions['visible_boxes']
        
        # Compute target offset
        target_offset = torch.zeros_like(pred_offset)
        target_offset[:, 0] = target_amodal[:, 0] - visible_boxes[:, 0]  # x1
        target_offset[:, 1] = target_amodal[:, 1] - visible_boxes[:, 1]  # y1
        target_offset[:, 2] = target_amodal[:, 2] - visible_boxes[:, 2]  # x2
        target_offset[:, 3] = target_amodal[:, 3] - visible_boxes[:, 3]  # y2
        
        # Offset L1 loss
        loss_offset = F.l1_loss(pred_offset, target_offset, reduction='mean')
        
        # GIoU loss
        giou = self.generalized_box_iou(pred_amodal, target_amodal)
        loss_giou = (1 - giou).mean()
        
        # Occlusion loss
        loss_occlusion = F.mse_loss(pred_occlusion, target_occlusion, reduction='mean')
        
        # Total loss
        total_loss = (
            self.weight_offset * loss_offset +
            self.weight_giou * loss_giou +
            self.weight_occlusion * loss_occlusion
        )
        
        return total_loss, {
            'loss_offset': loss_offset.item(),
            'loss_giou': loss_giou.item(),
            'loss_occlusion': loss_occlusion.item(),
            'total': total_loss.item(),
            'mean_giou': giou.mean().item(),
            'mean_iou': self.box_iou(pred_amodal, target_amodal).mean().item()
        }


def extract_roi_features(feature_map, boxes, roi_size=7):
    """
    Extract RoI features using RoIAlign
    
    Args:
        feature_map: [1, C, H, W] feature from backbone
        boxes: [N, 4] boxes [x1, y1, x2, y2] normalized [0, 1]
        roi_size: output spatial size
    
    Returns:
        [N, C, roi_size, roi_size] RoI features
    """
    if boxes.size(0) == 0:
        C = feature_map.size(1)
        return torch.zeros(0, C, roi_size, roi_size, device=boxes.device)
    
    # Convert normalized boxes to absolute coordinates
    H, W = feature_map.shape[2:]
    abs_boxes = boxes.clone()
    abs_boxes[:, [0, 2]] *= W
    abs_boxes[:, [1, 3]] *= H
    
    # Add batch index
    batch_indices = torch.zeros(len(boxes), 1, device=boxes.device, dtype=boxes.dtype)
    rois = torch.cat([batch_indices, abs_boxes], dim=1)
    
    # Extract RoI features
    roi_features = roi_align(
        feature_map,
        rois,
        output_size=(roi_size, roi_size),
        spatial_scale=1.0,
        aligned=True
    )
    
    return roi_features