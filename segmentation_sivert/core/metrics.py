#!/usr/bin/env python3
"""
Core Metrics for DFINE Segmentation
Evaluation metrics for segmentation performance
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class SegmentationMetrics:
    """Comprehensive segmentation metrics calculation"""
    
    def __init__(self, num_classes: int = 7, ignore_index: int = 255, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        
        # Default Pascal Person Parts class names
        if num_classes == 7 and class_names is None:
            self.class_names = [
                'background', 'head', 'torso', 'arms', 'hands', 'legs', 'feet'
            ]
        
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.pixel_counts = np.zeros(self.num_classes)
        self.total_pixels = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with new predictions and targets
        
        Args:
            pred: Predicted logits [N, C, H, W] or class predictions [N, H, W]
            target: Ground truth labels [N, H, W]
        """
        # Convert logits to class predictions if needed
        if pred.dim() == 4:  # [N, C, H, W]
            pred = torch.argmax(pred, dim=1)
        
        # Flatten tensors
        pred_flat = pred.flatten().cpu().numpy()
        target_flat = target.flatten().cpu().numpy()
        
        # Create mask for valid pixels
        valid_mask = target_flat != self.ignore_index
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        # Update confusion matrix
        for i in range(len(pred_valid)):
            if 0 <= target_valid[i] < self.num_classes and 0 <= pred_valid[i] < self.num_classes:
                self.confusion_matrix[target_valid[i], pred_valid[i]] += 1
                self.pixel_counts[target_valid[i]] += 1
                self.total_pixels += 1
    
    def compute_miou(self) -> float:
        """Compute mean Intersection over Union"""
        ious = []
        for i in range(self.num_classes):
            true_positive = self.confusion_matrix[i, i]
            false_positive = self.confusion_matrix[:, i].sum() - true_positive
            false_negative = self.confusion_matrix[i, :].sum() - true_positive
            
            union = true_positive + false_positive + false_negative
            if union > 0:
                iou = true_positive / union
                ious.append(iou)
        
        return np.mean(ious) if ious else 0.0
    
    def compute_class_iou(self) -> Dict[str, float]:
        """Compute per-class IoU"""
        class_ious = {}
        for i in range(self.num_classes):
            true_positive = self.confusion_matrix[i, i]
            false_positive = self.confusion_matrix[:, i].sum() - true_positive
            false_negative = self.confusion_matrix[i, :].sum() - true_positive
            
            union = true_positive + false_positive + false_negative
            if union > 0:
                iou = true_positive / union
            else:
                iou = 0.0
            
            class_ious[self.class_names[i]] = iou
        
        return class_ious
    
    def compute_pixel_accuracy(self) -> float:
        """Compute overall pixel accuracy"""
        correct_pixels = np.diag(self.confusion_matrix).sum()
        return correct_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
    
    def compute_class_accuracy(self) -> Dict[str, float]:
        """Compute per-class accuracy (recall)"""
        class_accuracies = {}
        for i in range(self.num_classes):
            true_positive = self.confusion_matrix[i, i]
            total_true = self.confusion_matrix[i, :].sum()
            
            if total_true > 0:
                accuracy = true_positive / total_true
            else:
                accuracy = 0.0
            
            class_accuracies[self.class_names[i]] = accuracy
        
        return class_accuracies
    
    def compute_class_precision(self) -> Dict[str, float]:
        """Compute per-class precision"""
        class_precisions = {}
        for i in range(self.num_classes):
            true_positive = self.confusion_matrix[i, i]
            total_pred = self.confusion_matrix[:, i].sum()
            
            if total_pred > 0:
                precision = true_positive / total_pred
            else:
                precision = 0.0
            
            class_precisions[self.class_names[i]] = precision
        
        return class_precisions
    
    def compute_f1_score(self) -> Dict[str, float]:
        """Compute per-class F1 score"""
        class_f1s = {}
        accuracies = self.compute_class_accuracy()
        precisions = self.compute_class_precision()
        
        for class_name in self.class_names:
            acc = accuracies[class_name]
            prec = precisions[class_name]
            
            if acc + prec > 0:
                f1 = 2 * (acc * prec) / (acc + prec)
            else:
                f1 = 0.0
            
            class_f1s[class_name] = f1
        
        return class_f1s
    
    def compute_frequency_weighted_iou(self) -> float:
        """Compute frequency weighted IoU"""
        class_ious = list(self.compute_class_iou().values())
        class_frequencies = self.pixel_counts / self.total_pixels if self.total_pixels > 0 else np.zeros(self.num_classes)
        
        fwiou = 0.0
        for i in range(self.num_classes):
            fwiou += class_frequencies[i] * class_ious[i]
        
        return fwiou
    
    def get_summary(self) -> Dict[str, float]:
        """Get comprehensive metrics summary"""
        summary = {
            'miou': self.compute_miou(),
            'pixel_accuracy': self.compute_pixel_accuracy(),
            'fwiou': self.compute_frequency_weighted_iou()
        }
        
        # Add per-class metrics
        class_ious = self.compute_class_iou()
        class_accs = self.compute_class_accuracy()
        class_precs = self.compute_class_precision()
        class_f1s = self.compute_f1_score()
        
        for class_name in self.class_names:
            summary[f'{class_name}_iou'] = class_ious[class_name]
            summary[f'{class_name}_acc'] = class_accs[class_name]
            summary[f'{class_name}_prec'] = class_precs[class_name]
            summary[f'{class_name}_f1'] = class_f1s[class_name]
        
        return summary
    
    def print_summary(self):
        """Print metrics summary"""
        summary = self.get_summary()
        
        print("📊 Segmentation Metrics Summary:")
        print(f"   mIoU: {summary['miou']:.4f}")
        print(f"   Pixel Accuracy: {summary['pixel_accuracy']:.4f}")
        print(f"   Frequency Weighted IoU: {summary['fwiou']:.4f}")
        
        print("\n📋 Per-Class Metrics:")
        for class_name in self.class_names:
            iou = summary[f'{class_name}_iou']
            acc = summary[f'{class_name}_acc']
            prec = summary[f'{class_name}_prec']
            f1 = summary[f'{class_name}_f1']
            print(f"   {class_name:>12}: IoU={iou:.3f}, Acc={acc:.3f}, Prec={prec:.3f}, F1={f1:.3f}")


def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 7, ignore_index: int = 255) -> float:
    """Quick mIoU computation for training loops"""
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        # Exclude ignore_index pixels
        if ignore_index is not None:
            valid_mask = (target != ignore_index)
            pred_cls = pred_cls & valid_mask
            target_cls = target_cls & valid_mask
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            ious.append(intersection / union)
    
    return torch.tensor(ious).mean().item() if ious else 0.0


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> float:
    """Quick pixel accuracy computation"""
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    if ignore_index is not None:
        valid_mask = (target != ignore_index)
        correct = (pred == target) & valid_mask
        total = valid_mask.sum()
    else:
        correct = (pred == target)
        total = torch.numel(pred)
    
    return (correct.sum().float() / total.float()).item() if total > 0 else 0.0


def compute_class_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 7, ignore_index: int = 255) -> List[float]:
    """Compute per-class IoU"""
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    class_ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        # Exclude ignore_index pixels
        if ignore_index is not None:
            valid_mask = (target != ignore_index)
            pred_cls = pred_cls & valid_mask
            target_cls = target_cls & valid_mask
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            class_ious.append((intersection / union).item())
        else:
            class_ious.append(0.0)
    
    return class_ious


class BatchMetricsTracker:
    """Track metrics across training batches"""
    
    def __init__(self, num_classes: int = 7, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset tracked metrics"""
        self.total_miou = 0.0
        self.total_pixel_acc = 0.0
        self.batch_count = 0
        self.class_ious = [0.0] * self.num_classes
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update with batch predictions"""
        miou = compute_miou(pred, target, self.num_classes, self.ignore_index)
        pixel_acc = compute_pixel_accuracy(pred, target, self.ignore_index)
        class_ious = compute_class_iou(pred, target, self.num_classes, self.ignore_index)
        
        self.total_miou += miou
        self.total_pixel_acc += pixel_acc
        self.batch_count += 1
        
        # Update class IoUs
        for i, iou in enumerate(class_ious):
            self.class_ious[i] += iou
    
    def get_averages(self) -> Dict[str, float]:
        """Get average metrics"""
        if self.batch_count == 0:
            return {'miou': 0.0, 'pixel_accuracy': 0.0}
        
        return {
            'miou': self.total_miou / self.batch_count,
            'pixel_accuracy': self.total_pixel_acc / self.batch_count,
            'class_ious': [iou / self.batch_count for iou in self.class_ious]
        }


# Factory function
def create_metrics_tracker(num_classes: int = 7, ignore_index: int = 255, class_names: Optional[List[str]] = None) -> SegmentationMetrics:
    """Create metrics tracker"""
    return SegmentationMetrics(num_classes, ignore_index, class_names)