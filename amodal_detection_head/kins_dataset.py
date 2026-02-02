#!/usr/bin/env python3
"""
KINS Dataset Loader for Amodal Bounding Box Detection
Loads both visible (inmodal) and amodal bounding boxes from KINS dataset
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pycocotools.coco import COCO
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class KINSAmodalDataset(Dataset):
    """
    KINS Dataset for Amodal Detection
    
    KINS provides:
    - Visible bounding boxes (i_bbox) - what you can see
    - Amodal bounding boxes (a_bbox) - full object extent
    - Occlusion information
    """
    
    # KINS categories (7 classes + background)
    CATEGORIES = {
        0: 'background',
        1: 'cyclist',
        2: 'pedestrian', 
        3: 'car',
        4: 'tram',
        5: 'truck',
        6: 'van',
        7: 'misc'
    }
    
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 image_size: int = 640,
                 max_objects: int = 100,
                 augment: bool = True):
        """
        Args:
            root_dir: Path to KINS dataset root
            split: 'train' or 'test'
            image_size: Target image size
            max_objects: Maximum number of objects per image (for padding)
            augment: Whether to apply augmentations (train only)
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.max_objects = max_objects
        
        # Normalize split naming
        split = 'val' if split in ['val', 'test', 'validation'] else split
        self.split = split

        # Paths
        self.img_dir = os.path.join(root_dir, 'training' if split == 'train' else 'testing', 'image_2')
        self.ann_file = os.path.join(root_dir, f'instances_{split}.json')
        
        # Load COCO-style annotations
        self.coco = COCO(self.ann_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        # Normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Augmentations (only for training)
        if augment and split == 'train':
            self.transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),
            ], bbox_params=A.BboxParams(
                format='coco',
                min_visibility=0.0
            ),
            additional_targets={'amodal_boxes': 'bboxes'})
        else:
            self.transforms = None
        
        print(f"📊 Loaded KINS {split} set:")
        print(f"   Images: {len(self.image_ids)}")
        print(f"   Categories: {len(self.CATEGORIES)}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def _calculate_occlusion_score(self, visible_bbox, amodal_bbox):
        """
        Calculate occlusion score as 1 - (visible_area / amodal_area)
        
        Args:
            visible_bbox: [x, y, w, h] in absolute coordinates
            amodal_bbox: [x, y, w, h] in absolute coordinates
        
        Returns:
            Occlusion score between 0 (not occluded) and 1 (fully occluded)
        """
        visible_area = visible_bbox[2] * visible_bbox[3]
        amodal_area = amodal_bbox[2] * amodal_bbox[3]
        
        if amodal_area == 0:
            return 0.0
        
        occlusion_score = 1.0 - (visible_area / amodal_area)
        return np.clip(occlusion_score, 0.0, 1.0)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index"""
        
        image_id = self.image_ids[idx]
        
        # Load image
        img_info = self.coco.imgs[image_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract boxes and labels
        visible_boxes = []
        amodal_boxes = []
        class_labels = []
        occlusion_scores = []
        
        valid_category_ids = set(self.CATEGORIES.keys())
        for ann in anns:
            # Visible (inmodal) box - marked with "i_bbox"
            if 'i_bbox' in ann:
                visible_bbox = ann['i_bbox']  # [x, y, w, h]
            else:
                visible_bbox = ann['bbox']  # Fallback
            
            # Amodal box - marked with "a_bbox"
            if 'a_bbox' in ann:
                amodal_bbox = ann['a_bbox']  # [x, y, w, h]
            else:
                amodal_bbox = visible_bbox  # If no amodal, assume no occlusion
            
            # Category
            category_id = ann['category_id']
            if category_id not in valid_category_ids:
                continue
            
            # Calculate occlusion
            occ_score = self._calculate_occlusion_score(visible_bbox, amodal_bbox)
            
            visible_boxes.append(visible_bbox)
            amodal_boxes.append(amodal_bbox)
            class_labels.append(category_id)
            occlusion_scores.append(occ_score)
        
        # Apply augmentations
        if self.transforms is not None and len(visible_boxes) > 0:
            transformed = self.transforms(
                image=image,
                bboxes=visible_boxes,
                amodal_boxes=amodal_boxes
            )
            image = transformed['image']
            visible_boxes = transformed['bboxes']
            amodal_boxes = transformed['amodal_boxes']
            occlusion_scores = [
                self._calculate_occlusion_score(vis_box, amod_box)
                for vis_box, amod_box in zip(visible_boxes, amodal_boxes)
            ]
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Convert boxes to normalized coordinates [0, 1] and [cx, cy, w, h] format
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        
        visible_boxes_norm = []
        amodal_boxes_norm = []
        
        for vis_box, amod_box in zip(visible_boxes, amodal_boxes):
            # Scale to resized image
            vis_x, vis_y, vis_w, vis_h = vis_box
            vis_x *= scale_x
            vis_y *= scale_y
            vis_w *= scale_x
            vis_h *= scale_y
            
            amod_x, amod_y, amod_w, amod_h = amod_box
            amod_x *= scale_x
            amod_y *= scale_y
            amod_w *= scale_x
            amod_h *= scale_y
            
            # Convert to center format and normalize
            vis_cx = (vis_x + vis_w / 2) / self.image_size
            vis_cy = (vis_y + vis_h / 2) / self.image_size
            vis_w_norm = vis_w / self.image_size
            vis_h_norm = vis_h / self.image_size
            
            amod_cx = (amod_x + amod_w / 2) / self.image_size
            amod_cy = (amod_y + amod_h / 2) / self.image_size
            amod_w_norm = amod_w / self.image_size
            amod_h_norm = amod_h / self.image_size
            
            visible_boxes_norm.append([vis_cx, vis_cy, vis_w_norm, vis_h_norm])
            amodal_boxes_norm.append([amod_cx, amod_cy, amod_w_norm, amod_h_norm])
        
        # Pad or truncate to max_objects
        num_objects = len(class_labels)
        
        if num_objects > self.max_objects:
            # Truncate
            visible_boxes_norm = visible_boxes_norm[:self.max_objects]
            amodal_boxes_norm = amodal_boxes_norm[:self.max_objects]
            class_labels = class_labels[:self.max_objects]
            occlusion_scores = occlusion_scores[:self.max_objects]
            num_objects = self.max_objects
        
        # Create tensors with padding
        visible_boxes_tensor = torch.zeros(self.max_objects, 4)
        amodal_boxes_tensor = torch.zeros(self.max_objects, 4)
        class_labels_tensor = torch.zeros(self.max_objects, dtype=torch.long)
        occlusion_tensor = torch.zeros(self.max_objects)
        valid_mask = torch.zeros(self.max_objects)
        
        if num_objects > 0:
            visible_boxes_tensor[:num_objects] = torch.tensor(visible_boxes_norm, dtype=torch.float32)
            amodal_boxes_tensor[:num_objects] = torch.tensor(amodal_boxes_norm, dtype=torch.float32)
            class_labels_tensor[:num_objects] = torch.tensor(class_labels, dtype=torch.long)
            occlusion_tensor[:num_objects] = torch.tensor(occlusion_scores, dtype=torch.float32)
            valid_mask[:num_objects] = 1.0
        
        return {
            'image': image,
            'visible_boxes': visible_boxes_tensor,
            'amodal_boxes': amodal_boxes_tensor,
            'class_labels': class_labels_tensor,
            'occlusion_scores': occlusion_tensor,
            'valid_mask': valid_mask,
            'image_id': image_id,
            'orig_size': torch.tensor([orig_h, orig_w])
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    images = torch.stack([item['image'] for item in batch])
    visible_boxes = torch.stack([item['visible_boxes'] for item in batch])
    amodal_boxes = torch.stack([item['amodal_boxes'] for item in batch])
    class_labels = torch.stack([item['class_labels'] for item in batch])
    occlusion_scores = torch.stack([item['occlusion_scores'] for item in batch])
    valid_mask = torch.stack([item['valid_mask'] for item in batch])
    
    return {
        'image': images,
        'visible_boxes': visible_boxes,
        'amodal_boxes': amodal_boxes,
        'class_labels': class_labels,
        'occlusion_scores': occlusion_scores,
        'valid_mask': valid_mask
    }


# Example usage
if __name__ == '__main__':
    # Test dataset loading
    dataset = KINSAmodalDataset(
        root_dir='/path/to/KINS',
        split='train',
        image_size=640
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading one sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Visible boxes shape: {sample['visible_boxes'].shape}")
    print(f"Amodal boxes shape: {sample['amodal_boxes'].shape}")
    print(f"Number of valid objects: {sample['valid_mask'].sum().item()}")
    
    # Show occlusion statistics
    valid_occ = sample['occlusion_scores'][sample['valid_mask'] > 0]
    print(f"\nOcclusion scores: min={valid_occ.min():.2f}, max={valid_occ.max():.2f}, mean={valid_occ.mean():.2f}")