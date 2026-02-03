#!/usr/bin/env python3
"""
Amodal Offset Prediction Head
MUCH BETTER APPROACH: Uses DFINE's person detections and predicts amodal offset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class AmodalOffsetHead(nn.Module):
    """
    Predicts amodal box offset from visible DFINE detections
    
    Input: DFINE visible boxes + RoI features
    Output: Amodal offset (how much to expand visible box)
    """
    
    def __init__(self, 
                 roi_size: int = 7,
                 in_channels: int = 256,
                 hidden_dim: int = 512):
        super().__init__()
        
        self.roi_size = roi_size
        
        # RoI feature processing
        self.roi_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Box embedding (encode visible box as feature)
        self.box_embed = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Fusion and prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # Predict offset from visible to amodal
        # Format: [delta_x1, delta_y1, delta_x2, delta_y2]
        # Negative values = expand left/up, positive = expand right/down
        self.offset_head = nn.Linear(hidden_dim // 2, 4)
        
        # Predict occlusion score
        self.occlusion_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Predict confidence in amodal prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, 
                roi_features: torch.Tensor,
                visible_boxes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            roi_features: [N, C, H, W] RoI features from DFINE
            visible_boxes: [N, 4] Visible boxes in [x1, y1, x2, y2] format
        
        Returns:
            Dict with amodal boxes, occlusion, confidence
        """
        # Process RoI features
        roi_feat = self.roi_conv(roi_features).flatten(1)  # [N, hidden_dim]
        
        # Encode visible box
        box_feat = self.box_embed(visible_boxes)  # [N, hidden_dim//2]
        
        # Fuse
        combined = torch.cat([roi_feat, box_feat], dim=1)
        features = self.fc(combined)
        
        # Predict offset
        offset = self.offset_head(features)  # [N, 4]
        
        # Compute amodal boxes from visible + offset
        # offset: [delta_x1, delta_y1, delta_x2, delta_y2]
        amodal_boxes = visible_boxes.clone()
        amodal_boxes[:, 0] = visible_boxes[:, 0] + offset[:, 0]  # x1
        amodal_boxes[:, 1] = visible_boxes[:, 1] + offset[:, 1]  # y1
        amodal_boxes[:, 2] = visible_boxes[:, 2] + offset[:, 2]  # x2
        amodal_boxes[:, 3] = visible_boxes[:, 3] + offset[:, 3]  # y2
        
        # Predict occlusion and confidence
        occlusion = self.occlusion_head(features)
        confidence = self.confidence_head(features)
        
        return {
            'amodal_boxes': amodal_boxes,
            'visible_boxes': visible_boxes,
            'offset': offset,
            'occlusion_scores': occlusion,
            'confidence_scores': confidence
        }


class AmodalOffsetLoss(nn.Module):
    """Loss for amodal offset prediction"""
    
    def __init__(self,
                 weight_offset: float = 10.0,
                 weight_giou: float = 5.0,
                 weight_occlusion: float = 2.0):
        super().__init__()
        self.weight_offset = weight_offset
        self.weight_giou = weight_giou
        self.weight_occlusion = weight_occlusion
    
    def compute_giou(self, boxes1, boxes2):
        """Compute GIoU between two sets of boxes [x1,y1,x2,y2]"""
        # Intersection
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        # Union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - inter
        
        iou = inter / union.clamp(min=1e-6)
        
        # Enclosing box
        lt_enc = torch.min(boxes1[:, :2], boxes2[:, :2])
        rb_enc = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        wh_enc = (rb_enc - lt_enc).clamp(min=0)
        area_enc = wh_enc[:, 0] * wh_enc[:, 1]
        
        giou = iou - (area_enc - union) / area_enc.clamp(min=1e-6)
        return giou
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with amodal_boxes, offset, occlusion_scores
            targets: dict with target_amodal_boxes, target_occlusion
        """
        pred_amodal = predictions['amodal_boxes']
        pred_offset = predictions['offset']
        pred_occlusion = predictions['occlusion_scores']
        
        target_amodal = targets['amodal_boxes']
        target_occlusion = targets['occlusion_scores']
        visible_boxes = predictions['visible_boxes']
        
        # Compute target offset
        target_offset = torch.zeros_like(pred_offset)
        target_offset[:, 0] = target_amodal[:, 0] - visible_boxes[:, 0]  # delta_x1
        target_offset[:, 1] = target_amodal[:, 1] - visible_boxes[:, 1]  # delta_y1
        target_offset[:, 2] = target_amodal[:, 2] - visible_boxes[:, 2]  # delta_x2
        target_offset[:, 3] = target_amodal[:, 3] - visible_boxes[:, 3]  # delta_y2
        
        # Offset loss (L1)
        loss_offset = F.l1_loss(pred_offset, target_offset)
        
        # GIoU loss
        giou = self.compute_giou(pred_amodal, target_amodal)
        loss_giou = (1 - giou).mean()
        
        # Occlusion loss
        loss_occlusion = F.mse_loss(pred_occlusion.squeeze(), target_occlusion)
        
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
            'mean_giou': giou.mean().item()
        }


def extract_roi_features(feature_maps, boxes, roi_size=7):
    """
    Extract RoI features using RoIAlign
    
    Args:
        feature_maps: [1, C, H, W] feature map from backbone
        boxes: [N, 4] boxes in [x1, y1, x2, y2] format, normalized [0, 1]
        roi_size: output size
    
    Returns:
        [N, C, roi_size, roi_size] RoI features
    """
    from torchvision.ops import roi_align
    
    # Convert normalized boxes to absolute coordinates
    H, W = feature_maps.shape[2:]
    abs_boxes = boxes.clone()
    abs_boxes[:, [0, 2]] *= W
    abs_boxes[:, [1, 3]] *= H
    
    # Add batch index (all boxes from same image)
    batch_indices = torch.zeros(len(boxes), 1, device=boxes.device)
    rois = torch.cat([batch_indices, abs_boxes], dim=1)
    
    # Extract RoI features
    roi_features = roi_align(
        feature_maps,
        rois,
        output_size=(roi_size, roi_size),
        spatial_scale=1.0,
        aligned=True
    )
    
    return roi_features