#!/usr/bin/env python3
"""
Amodal Detection Head for DFINE
Adds amodal bounding box prediction capability to existing DFINE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class AmodalDetectionHead(nn.Module):
    """
    Amodal Detection Head that predicts full bounding boxes for occluded objects.
    
    Takes DFINE features and predicts:
    - Visible bounding boxes (inmodal)
    - Amodal bounding boxes (full extent including occluded parts)
    - Occlusion score (how much is occluded)
    """
    
    def __init__(self, 
                 in_channels: int = 384,
                 hidden_dim: int = 256,
                 num_classes: int = 2,  # KINS categories
                 num_queries: int = 100):  # Number of detection queries
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Query embeddings for amodal detection
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer decoder for amodal box prediction
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Prediction heads
        # 1. Visible (inmodal) box prediction
        self.visible_bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4)  # x, y, w, h
        )
        
        # 2. Amodal box prediction
        self.amodal_bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4)  # x, y, w, h
        )
        
        # 3. Occlusion score (0 = not occluded, 1 = fully occluded)
        self.occlusion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 4. Class prediction for KINS categories
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        
        # 5. Confidence score
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
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
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            features: List of feature maps from DFINE backbone
                     [feat1, feat2, feat3, feat4] with different scales
        
        Returns:
            Dictionary with:
                - visible_boxes: [B, num_queries, 4]
                - amodal_boxes: [B, num_queries, 4]
                - occlusion_scores: [B, num_queries, 1]
                - class_logits: [B, num_queries, num_classes+1]
                - confidence_scores: [B, num_queries, 1]
        """
        # Use richest (last) backbone feature for amodal prediction
        feat = features[-1]
        B = feat.shape[0]
        
        # Project features
        feat = self.input_proj(feat)  # [B, hidden_dim, H, W]
        
        # Flatten spatial dimensions for transformer
        H, W = feat.shape[2:]
        feat_flat = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]
        
        # Get query embeddings
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, hidden_dim]
        
        # Apply transformer decoder
        # Queries attend to feature maps to predict amodal boxes
        memory = feat_flat
        decoded = self.transformer_decoder(queries, memory)  # [B, num_queries, hidden_dim]
        
        # Predict outputs
        visible_boxes = self.visible_bbox_head(decoded)  # [B, num_queries, 4]
        amodal_boxes = self.amodal_bbox_head(decoded)   # [B, num_queries, 4]
        occlusion_scores = self.occlusion_head(decoded)  # [B, num_queries, 1]
        class_logits = self.class_head(decoded)          # [B, num_queries, num_classes+1]
        confidence_scores = self.confidence_head(decoded) # [B, num_queries, 1]
        
        # Convert to absolute coordinates (assuming input is normalized)
        visible_boxes = torch.sigmoid(visible_boxes)
        amodal_boxes = torch.sigmoid(amodal_boxes)
        
        return {
            'visible_boxes': visible_boxes,
            'amodal_boxes': amodal_boxes,
            'occlusion_scores': occlusion_scores,
            'class_logits': class_logits,
            'confidence_scores': confidence_scores
        }


class AmodalLoss(nn.Module):
    """
    Loss function for amodal detection training
    
    Combines:
    - L1 loss for visible and amodal boxes
    - GIoU loss for box quality
    - Classification loss
    - Occlusion prediction loss
    """
    
    def __init__(self, 
                 weight_visible_l1: float = 5.0,
                 weight_amodal_l1: float = 5.0,
                 weight_visible_giou: float = 2.0,
                 weight_amodal_giou: float = 2.0,
                 weight_class: float = 1.0,
                 weight_occlusion: float = 1.0):
        super().__init__()
        
        self.weight_visible_l1 = weight_visible_l1
        self.weight_amodal_l1 = weight_amodal_l1
        self.weight_visible_giou = weight_visible_giou
        self.weight_amodal_giou = weight_amodal_giou
        self.weight_class = weight_class
        self.weight_occlusion = weight_occlusion
    
    def box_l1_loss(self, pred_boxes, target_boxes, valid_mask):
        """L1 loss for bounding boxes"""
        loss = F.l1_loss(pred_boxes, target_boxes, reduction='none')
        loss = (loss * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum().clamp(min=1)
        return loss
    
    def generalized_box_iou(self, boxes1, boxes2):
        """
        Compute Generalized IoU
        boxes in format [x, y, w, h] normalized to [0, 1]
        """
        # Convert to [x1, y1, x2, y2]
        boxes1_xyxy = torch.zeros_like(boxes1)
        boxes1_xyxy[..., 0] = boxes1[..., 0] - boxes1[..., 2] / 2  # x1
        boxes1_xyxy[..., 1] = boxes1[..., 1] - boxes1[..., 3] / 2  # y1
        boxes1_xyxy[..., 2] = boxes1[..., 0] + boxes1[..., 2] / 2  # x2
        boxes1_xyxy[..., 3] = boxes1[..., 1] + boxes1[..., 3] / 2  # y2
        
        boxes2_xyxy = torch.zeros_like(boxes2)
        boxes2_xyxy[..., 0] = boxes2[..., 0] - boxes2[..., 2] / 2
        boxes2_xyxy[..., 1] = boxes2[..., 1] - boxes2[..., 3] / 2
        boxes2_xyxy[..., 2] = boxes2[..., 0] + boxes2[..., 2] / 2
        boxes2_xyxy[..., 3] = boxes2[..., 1] + boxes2[..., 3] / 2
        
        # Calculate intersection
        lt = torch.max(boxes1_xyxy[..., :2], boxes2_xyxy[..., :2])
        rb = torch.min(boxes1_xyxy[..., 2:], boxes2_xyxy[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        
        # Calculate union
        area1 = (boxes1_xyxy[..., 2] - boxes1_xyxy[..., 0]) * (boxes1_xyxy[..., 3] - boxes1_xyxy[..., 1])
        area2 = (boxes2_xyxy[..., 2] - boxes2_xyxy[..., 0]) * (boxes2_xyxy[..., 3] - boxes2_xyxy[..., 1])
        union = area1 + area2 - inter
        
        iou = inter / union.clamp(min=1e-6)
        
        # Calculate enclosing box
        lt_enc = torch.min(boxes1_xyxy[..., :2], boxes2_xyxy[..., :2])
        rb_enc = torch.max(boxes1_xyxy[..., 2:], boxes2_xyxy[..., 2:])
        wh_enc = (rb_enc - lt_enc).clamp(min=0)
        area_enc = wh_enc[..., 0] * wh_enc[..., 1]
        
        # GIoU
        giou = iou - (area_enc - union) / area_enc.clamp(min=1e-6)
        
        return giou
    
    def forward(self, predictions, targets):
        """
        Compute loss
        
        Args:
            predictions: Dict from AmodalDetectionHead
            targets: Dict with:
                - visible_boxes: [B, N, 4]
                - amodal_boxes: [B, N, 4]
                - class_labels: [B, N]
                - occlusion_scores: [B, N]
                - valid_mask: [B, N] (1 for valid objects, 0 for padding)
        """
        pred_visible = predictions['visible_boxes']
        pred_amodal = predictions['amodal_boxes']
        pred_class = predictions['class_logits']
        pred_occlusion = predictions['occlusion_scores']
        
        tgt_visible = targets['visible_boxes']
        tgt_amodal = targets['amodal_boxes']
        tgt_class = targets['class_labels']
        tgt_occlusion = targets['occlusion_scores']
        valid_mask = targets['valid_mask']

        # Align prediction/target lengths
        pred_len = pred_visible.shape[1]
        tgt_len = tgt_visible.shape[1]
        if pred_len != tgt_len:
            common_len = min(pred_len, tgt_len)
            pred_visible = pred_visible[:, :common_len]
            pred_amodal = pred_amodal[:, :common_len]
            pred_class = pred_class[:, :common_len]
            pred_occlusion = pred_occlusion[:, :common_len]
            tgt_visible = tgt_visible[:, :common_len]
            tgt_amodal = tgt_amodal[:, :common_len]
            tgt_class = tgt_class[:, :common_len]
            tgt_occlusion = tgt_occlusion[:, :common_len]
            valid_mask = valid_mask[:, :common_len]
        
        # L1 losses
        loss_visible_l1 = self.box_l1_loss(pred_visible, tgt_visible, valid_mask)
        loss_amodal_l1 = self.box_l1_loss(pred_amodal, tgt_amodal, valid_mask)
        
        # GIoU losses
        giou_visible = self.generalized_box_iou(pred_visible, tgt_visible)
        giou_amodal = self.generalized_box_iou(pred_amodal, tgt_amodal)
        loss_visible_giou = (1 - giou_visible) * valid_mask
        loss_amodal_giou = (1 - giou_amodal) * valid_mask
        loss_visible_giou = loss_visible_giou.sum() / valid_mask.sum().clamp(min=1)
        loss_amodal_giou = loss_amodal_giou.sum() / valid_mask.sum().clamp(min=1)
        
        # Classification loss
        loss_class = F.cross_entropy(
            pred_class.reshape(-1, pred_class.shape[-1]),
            tgt_class.reshape(-1),
            reduction='none'
        )
        loss_class = (loss_class.reshape(valid_mask.shape) * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        
        # Occlusion loss
        loss_occlusion = F.mse_loss(
            pred_occlusion.squeeze(-1),
            tgt_occlusion,
            reduction='none'
        )
        loss_occlusion = (loss_occlusion * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        
        # Total loss
        total_loss = (
            self.weight_visible_l1 * loss_visible_l1 +
            self.weight_amodal_l1 * loss_amodal_l1 +
            self.weight_visible_giou * loss_visible_giou +
            self.weight_amodal_giou * loss_amodal_giou +
            self.weight_class * loss_class +
            self.weight_occlusion * loss_occlusion
        )
        
        return total_loss, {
            'loss_visible_l1': loss_visible_l1.item(),
            'loss_amodal_l1': loss_amodal_l1.item(),
            'loss_visible_giou': loss_visible_giou.item(),
            'loss_amodal_giou': loss_amodal_giou.item(),
            'loss_class': loss_class.item(),
            'loss_occlusion': loss_occlusion.item(),
            'total': total_loss.item()
        }


class CombinedDFINEAmodalModel(nn.Module):
    """
    Combined model with:
    - DFINE detection (frozen)
    - Pascal segmentation head (frozen)
    - NEW: Amodal detection head (trainable)
    """
    
    def __init__(self,
                 dfine_model: nn.Module,
                 segmentation_head: nn.Module,
                 amodal_head: AmodalDetectionHead):
        super().__init__()
        
        self.dfine_model = dfine_model
        self.segmentation_head = segmentation_head
        self.amodal_head = amodal_head
        
        # Freeze DFINE and segmentation
        for param in self.dfine_model.parameters():
            param.requires_grad = False
        for param in self.segmentation_head.parameters():
            param.requires_grad = False
        
        print("🔒 Frozen DFINE detection and segmentation heads")
        print("🎯 Training only amodal detection head")
    
    def forward(self, x: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through all components"""
        
        # Get backbone features
        with torch.no_grad():
            backbone_features = self.dfine_model.backbone(x)
        
        outputs = {}
        
        # DFINE detection (frozen, optional)
        if targets is not None:
            with torch.no_grad():
                det_outputs = self.dfine_model(x, targets=targets)
                outputs['detection'] = det_outputs
        
        # Segmentation (frozen, for reference)
        with torch.no_grad():
            seg_logits = self.segmentation_head(backbone_features)
            seg_logits = F.interpolate(
                seg_logits, size=x.shape[-2:],
                mode='bilinear', align_corners=False
            )
            outputs['segmentation'] = seg_logits
        
        # Amodal detection (trainable)
        amodal_outputs = self.amodal_head(backbone_features)
        outputs.update(amodal_outputs)
        
        return outputs


# Example usage
def create_amodal_model(dfine_model, segmentation_head):
    """Factory function to create combined model"""
    
    # Determine feature channels from backbone
    # For HGNetv2, typically [64, 128, 256, 384]
    in_channels = 384  # Use the richest features
    
    amodal_head = AmodalDetectionHead(
        in_channels=in_channels,
        hidden_dim=256,
        num_classes=2,  # COCOA: background + person
        num_queries=100
    )
    
    model = CombinedDFINEAmodalModel(
        dfine_model=dfine_model,
        segmentation_head=segmentation_head,
        amodal_head=amodal_head
    )
    
    return model, AmodalLoss()