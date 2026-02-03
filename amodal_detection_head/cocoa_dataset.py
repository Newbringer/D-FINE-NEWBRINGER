#!/usr/bin/env python3
"""
COCOA Dataset - Humans Only, Occlusion-focused
Clean implementation for amodal detection training
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
from typing import Dict, List
import albumentations as A


# Complete list of INDIVIDUAL human categories from COCOA
# Excludes: crowds, groups, audiences, body parts
HUMAN_CATEGORIES = {
    # Basic humans
    'man', 'Man', 'men', 'Men',
    'woman', 'Woman', 'women', 'Women', 'weman', 'Weoman', 'wwoman',
    'boy', 'Boy', 'Boys', 'Bboy',
    'girl', 'Girl', 'girls',
    'child', 'Child', 'children', 'Chlldren', 'chlid',
    'baby', 'Baby', 'boby',
    'people', 'People', 'peoples', 'Peoples',
    'person', 'Person',
    'Lady',
    'Kid', 'kid',
    'Old Man',
    'Old Woman',
    'the man',
    'mom',
    
    # Roles/Occupations/Activities
    'player', 'Player',
    'athlete',
    'waiter',
    'plumber',
    'skater',
    'Ice Skating', 'ice skating',
    'skiier',
    'model', 'models',
    'supporter',
    'timekeeper',
    'sitter',
    'Camera man',
    'Snow Surfing Boy',
}


class COCOAAmodalDataset(Dataset):
    """
    COCOA Dataset for Amodal Human Detection
    - Filters strictly for individual human categories
    - Trains only on occluded samples (min_occlusion > 0)
    - Returns visible + amodal boxes for offset prediction
    """
    
    def __init__(self,
                 image_dir: str,
                 ann_file: str,
                 split: str = 'train',
                 image_size: int = 640,
                 max_objects: int = 50,
                 augment: bool = True,
                 min_occlusion: float = 0.05,
                 min_area: int = 400):
        """
        Args:
            image_dir: Path to COCO images (train2014/val2014)
            ann_file: Path to COCOA annotation JSON
            min_occlusion: Minimum occlusion rate (0.05 = slightly occluded)
            min_area: Minimum box area in pixels after resize
        """
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.split = split
        self.image_size = image_size
        self.max_objects = max_objects
        self.min_occlusion = min_occlusion
        self.min_area = min_area
        
        print(f"\n{'='*80}")
        print(f"📂 Loading COCOA {split} - INDIVIDUAL HUMANS ONLY")
        print(f"{'='*80}")
        print(f"Annotation file: {ann_file}")
        print(f"Min occlusion: {min_occlusion:.2f}")
        print(f"Min area: {min_area}px²")
        
        # Load annotations
        with open(ann_file, 'r') as f:
            self.data = json.load(f)
        
        # Build dataset
        self._build_dataset()
        
        # Normalization (ImageNet stats)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Augmentations
        if augment and split == 'train':
            self.transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
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
    
    def _polygon_to_bbox(self, polygon_coords: List[float]) -> List[float]:
        """Convert polygon coordinates to [x, y, w, h] bbox"""
        if not polygon_coords or len(polygon_coords) < 6:
            return None
        
        try:
            xs = [polygon_coords[i] for i in range(0, len(polygon_coords), 2)]
            ys = [polygon_coords[i] for i in range(1, len(polygon_coords), 2)]
            
            x_min = max(0, min(xs))
            y_min = max(0, min(ys))
            x_max = max(xs)
            y_max = max(ys)
            
            w = x_max - x_min
            h = y_max - y_min
            
            if w <= 0 or h <= 0:
                return None
            
            return [float(x_min), float(y_min), float(w), float(h)]
        except:
            return None
    
    def _decode_rle_to_bbox(self, rle_mask: dict, img_width: int, img_height: int) -> List[float]:
        """Decode RLE mask to bbox [x, y, w, h]"""
        try:
            from pycocotools import mask as mask_util
            
            rle = {
                'counts': rle_mask['counts'].encode('utf-8') if isinstance(rle_mask['counts'], str) else rle_mask['counts'],
                'size': rle_mask['size']
            }
            
            binary_mask = mask_util.decode(rle)
            
            # Find bbox from mask
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            
            if not rows.any() or not cols.any():
                return None
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
        except:
            return None
    
    def _build_dataset(self):
        """Build dataset from COCOA annotations"""
        self.valid_images = []
        self.img_to_anns = {}
        self.total_humans = 0
        
        # Build image lookup
        img_id_to_info = {img['id']: img for img in self.data.get('images', [])}
        
        # Process annotations
        for ann in self.data.get('annotations', []):
            img_id = ann.get('image_id')
            if img_id not in img_id_to_info:
                continue
            
            img_info = img_id_to_info[img_id]
            img_width = img_info['width']
            img_height = img_info['height']
            
            regions = ann.get('regions', [])
            human_regions = []
            
            for region in regions:
                name = region.get('name', '').strip()
                
                # Filter: only individual humans
                if name not in HUMAN_CATEGORIES:
                    continue
                
                # Get occlusion rate
                occlude_rate = region.get('occlude_rate', 0)
                if occlude_rate is None:
                    occlude_rate = 0
                occlude_rate = float(occlude_rate)
                occlude_rate = max(0.0, min(1.0, occlude_rate))
                
                # Filter: only occluded samples
                if occlude_rate < self.min_occlusion:
                    continue
                
                # Get amodal bbox from full segmentation
                segmentation = region.get('segmentation', [])
                amodal_bbox = self._polygon_to_bbox(segmentation)
                if amodal_bbox is None:
                    continue
                
                # Get visible bbox from visible_mask (if available)
                visible_mask = region.get('visible_mask')
                if visible_mask:
                    visible_bbox = self._decode_rle_to_bbox(visible_mask, img_width, img_height)
                    if visible_bbox is None:
                        # Fallback: estimate from occlusion
                        scale = np.sqrt(1.0 - occlude_rate)
                        x, y, w, h = amodal_bbox
                        cx, cy = x + w/2, y + h/2
                        new_w, new_h = w * scale, h * scale
                        visible_bbox = [cx - new_w/2, cy - new_h/2, new_w, new_h]
                else:
                    # No visible mask: estimate from occlusion
                    scale = np.sqrt(1.0 - occlude_rate)
                    x, y, w, h = amodal_bbox
                    cx, cy = x + w/2, y + h/2
                    new_w, new_h = w * scale, h * scale
                    visible_bbox = [cx - new_w/2, cy - new_h/2, new_w, new_h]
                
                human_regions.append({
                    'visible_bbox': visible_bbox,
                    'amodal_bbox': amodal_bbox,
                    'occlude_rate': occlude_rate,
                    'category': name
                })
            
            # Add image if it has human annotations
            if len(human_regions) > 0:
                if img_id not in self.img_to_anns:
                    self.valid_images.append({
                        'image_id': img_id,
                        'file_name': img_info['file_name'],
                        'width': img_info['width'],
                        'height': img_info['height']
                    })
                    self.img_to_anns[img_id] = []
                
                self.img_to_anns[img_id].extend(human_regions)
                self.total_humans += len(human_regions)
    
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
        anns = self.img_to_anns[image_id]
        
        visible_boxes = []
        amodal_boxes = []
        occlusion_scores = []
        class_labels = []
        
        for ann in anns:
            visible_boxes.append(ann['visible_bbox'])
            amodal_boxes.append(ann['amodal_bbox'])
            occlusion_scores.append(ann['occlude_rate'])
            class_labels.append(1)  # All humans = class 1
        
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
                
                # Keep same number of amodal boxes
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
        
        # Convert boxes to normalized [x1, y1, x2, y2] format
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        
        visible_boxes_norm = []
        amodal_boxes_norm = []
        valid_occlusion = []
        
        for vis_box, amod_box, occ in zip(visible_boxes, amodal_boxes, occlusion_scores):
            # Scale visible box
            vis_x, vis_y, vis_w, vis_h = vis_box
            vis_x1 = vis_x * scale_x
            vis_y1 = vis_y * scale_y
            vis_x2 = (vis_x + vis_w) * scale_x
            vis_y2 = (vis_y + vis_h) * scale_y
            
            # Scale amodal box
            amod_x, amod_y, amod_w, amod_h = amod_box
            amod_x1 = amod_x * scale_x
            amod_y1 = amod_y * scale_y
            amod_x2 = (amod_x + amod_w) * scale_x
            amod_y2 = (amod_y + amod_h) * scale_y
            
            # Filter by minimum area
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
        
        # Pad or truncate to max_objects
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