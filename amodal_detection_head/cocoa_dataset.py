#!/usr/bin/env python3
"""
Synthetic Amodal Dataset from COCO
Replaces COCOA dataset with controlled synthetic occlusions
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
from typing import Dict
import albumentations as A


class COCOAAmodalDataset(Dataset):
    """
    Synthetic Amodal Dataset - loads from JSON annotations
    Drop-in replacement for original COCOA loader
    """
    
    def __init__(self,
                 image_dir: str,
                 ann_file: str,
                 split: str = 'train',
                 image_size: int = 640,
                 max_objects: int = 50,
                 augment: bool = True,
                 min_occlusion: float = 0.10,
                 min_area: int = 200):
        """
        Args:
            image_dir: Path to COCO images (train2014/val2014)
            ann_file: Path to synthetic_amodal_annotations.json
            split: 'train' or 'val'
            image_size: Resize size
            max_objects: Max objects per image
            augment: Apply augmentations
            min_occlusion: Minimum occlusion rate
            min_area: Minimum box area in pixels
        """
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.split = split
        self.image_size = image_size
        self.max_objects = max_objects
        self.min_occlusion = min_occlusion
        self.min_area = min_area
        
        print(f"\n{'='*80}")
        print(f"📂 Loading Synthetic Amodal Dataset - {split.upper()}")
        print(f"{'='*80}")
        print(f"Annotation file: {ann_file}")
        print(f"Min occlusion: {min_occlusion:.2f}")
        print(f"Min area: {min_area}px²")
        
        # Load annotations
        with open(ann_file, 'r') as f:
            self.data = json.load(f)
        
        # Filter by occlusion
        self.data = [
            d for d in self.data 
            if d['occlusion_rate'] >= min_occlusion
        ]
        
        # Group by image
        self._group_by_image()
        
        # Normalization (ImageNet stats)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Augmentations
        if augment and split == 'train':
            self.transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.3
                ),
            ], bbox_params=A.BboxParams(
                format='coco',
                min_visibility=0.3,
                label_fields=['class_labels']
            ))
        else:
            self.transforms = None
        
        print(f"\n✅ Dataset ready:")
        print(f"   Images: {len(self.valid_images)}")
        print(f"   Human annotations: {self.total_humans}")
        print(f"   Avg humans/image: {self.total_humans / max(len(self.valid_images), 1):.2f}")
        print(f"{'='*80}\n")
    
    def _group_by_image(self):
        """Group annotations by image"""
        self.img_to_anns = {}
        
        for d in self.data:
            img_id = d['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = {
                    'file_name': d['file_name'],
                    'width': d['image_width'],
                    'height': d['image_height'],
                    'annotations': []
                }
            
            self.img_to_anns[img_id]['annotations'].append({
                'visible_bbox': d['visible_bbox'],
                'amodal_bbox': d['amodal_bbox'],
                'occlude_rate': d['occlusion_rate'],
                'category': d['category']
            })
        
        self.valid_images = [
            {'image_id': img_id, **info}
            for img_id, info in self.img_to_anns.items()
        ]
        
        self.total_humans = sum(
            len(info['annotations']) 
            for info in self.img_to_anns.values()
        )
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_info = self.valid_images[idx]
        image_id = img_info['image_id']
        file_name = img_info['file_name']
        
        # Load image
        img_path = os.path.join(self.image_dir, file_name)
        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Get annotations
        anns = self.img_to_anns[image_id]['annotations']
        
        visible_boxes = []
        amodal_boxes = []
        occlusion_scores = []
        class_labels = []
        
        for ann in anns:
            visible_boxes.append(ann['visible_bbox'])
            amodal_boxes.append(ann['amodal_bbox'])
            occlusion_scores.append(ann['occlude_rate'])
            class_labels.append(1)
        
        if len(visible_boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))
        
        # Apply augmentations
        if self.transforms is not None:
            try:
                transformed = self.transforms(
                    image=image,
                    bboxes=visible_boxes,
                    class_labels=class_labels
                )
                image = transformed['image']
                visible_boxes = list(transformed['bboxes'])
                class_labels = transformed['class_labels']
                
                if len(visible_boxes) < len(amodal_boxes):
                    amodal_boxes = amodal_boxes[:len(visible_boxes)]
                    occlusion_scores = occlusion_scores[:len(visible_boxes)]
            except:
                pass
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Convert boxes to normalized [x1, y1, x2, y2]
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        
        visible_boxes_norm = []
        amodal_boxes_norm = []
        valid_occlusion = []
        
        for vis_box, amod_box, occ in zip(visible_boxes, amodal_boxes, occlusion_scores):
            # Scale boxes
            vis_x, vis_y, vis_w, vis_h = vis_box
            vis_x1 = vis_x * scale_x
            vis_y1 = vis_y * scale_y
            vis_x2 = (vis_x + vis_w) * scale_x
            vis_y2 = (vis_y + vis_h) * scale_y
            
            amod_x, amod_y, amod_w, amod_h = amod_box
            amod_x1 = amod_x * scale_x
            amod_y1 = amod_y * scale_y
            amod_x2 = (amod_x + amod_w) * scale_x
            amod_y2 = (amod_y + amod_h) * scale_y
            
            # Filter by area
            vis_area = (vis_x2 - vis_x1) * (vis_y2 - vis_y1)
            amod_area = (amod_x2 - amod_x1) * (amod_y2 - amod_y1)
            
            if vis_area < self.min_area or amod_area < self.min_area:
                continue
            
            # Normalize to [0, 1]
            vis_x1_norm = np.clip(vis_x1 / self.image_size, 0, 1)
            vis_y1_norm = np.clip(vis_y1 / self.image_size, 0, 1)
            vis_x2_norm = np.clip(vis_x2 / self.image_size, 0, 1)
            vis_y2_norm = np.clip(vis_y2 / self.image_size, 0, 1)
            
            amod_x1_norm = np.clip(amod_x1 / self.image_size, 0, 1)
            amod_y1_norm = np.clip(amod_y1 / self.image_size, 0, 1)
            amod_x2_norm = np.clip(amod_x2 / self.image_size, 0, 1)
            amod_y2_norm = np.clip(amod_y2 / self.image_size, 0, 1)
            
            visible_boxes_norm.append([vis_x1_norm, vis_y1_norm, vis_x2_norm, vis_y2_norm])
            amodal_boxes_norm.append([amod_x1_norm, amod_y1_norm, amod_x2_norm, amod_y2_norm])
            valid_occlusion.append(occ)
        
        # Pad or truncate
        num_objects = len(visible_boxes_norm)
        
        if num_objects > self.max_objects:
            visible_boxes_norm = visible_boxes_norm[:self.max_objects]
            amodal_boxes_norm = amodal_boxes_norm[:self.max_objects]
            valid_occlusion = valid_occlusion[:self.max_objects]
            num_objects = self.max_objects
        
        # Create tensors
        visible_boxes_tensor = torch.zeros(self.max_objects, 4)
        amodal_boxes_tensor = torch.zeros(self.max_objects, 4)
        occlusion_tensor = torch.zeros(self.max_objects)
        valid_mask = torch.zeros(self.max_objects)
        
        if num_objects > 0:
            visible_boxes_tensor[:num_objects] = torch.tensor(visible_boxes_norm, dtype=torch.float32)
            amodal_boxes_tensor[:num_objects] = torch.tensor(amodal_boxes_norm, dtype=torch.float32)
            occlusion_tensor[:num_objects] = torch.tensor(valid_occlusion, dtype=torch.float32)
            valid_mask[:num_objects] = 1.0
        
        return {
            'image': image,
            'visible_boxes': visible_boxes_tensor,
            'amodal_boxes': amodal_boxes_tensor,
            'occlusion_scores': occlusion_tensor,
            'valid_mask': valid_mask
        }


def collate_fn(batch):
    """Custom collate function"""
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'visible_boxes': torch.stack([item['visible_boxes'] for item in batch]),
        'amodal_boxes': torch.stack([item['amodal_boxes'] for item in batch]),
        'occlusion_scores': torch.stack([item['occlusion_scores'] for item in batch]),
        'valid_mask': torch.stack([item['valid_mask'] for item in batch])
    }