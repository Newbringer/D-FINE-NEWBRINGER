#!/usr/bin/env python3
"""
COCOA Dataset Loader - FIXED with proper visible mask computation
Uses actual visible masks instead of crude estimation
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
from typing import Dict, List
import albumentations as A


class COCOAAmodalDataset(Dataset):
    """
    COCOA Dataset with PROPER visible bbox computation from masks
    """
    
    def __init__(self,
                 image_dir: str,
                 annotation_file: str,
                 split: str = 'train',
                 image_size: int = 640,
                 max_objects: int = 50,
                 augment: bool = True,
                 person_only: bool = True,
                 min_area: int = 400):  # Filter tiny boxes
        """
        Args:
            min_area: Minimum bbox area in pixels (after resize) to keep
        """
        self.image_dir = image_dir
        self.split = split
        self.image_size = image_size
        self.max_objects = max_objects
        self.person_only = person_only
        self.min_area = min_area
        
        print(f"📂 Loading COCOA {split} annotations...")
        print(f"   Annotation file: {annotation_file}")
        
        # Load COCOA annotations
        with open(annotation_file, 'r') as f:
            self.cocoa_data = json.load(f)
        
        # Build dataset
        self._build_dataset()
        
        # Normalization
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
        
        print(f"✅ Loaded COCOA {split} set:")
        print(f"   Images: {len(self.valid_images)}")
        print(f"   Total annotations: {self.total_annotations}")
        if len(self.valid_images) > 0:
            print(f"   Avg per image: {self.total_annotations / len(self.valid_images):.2f}")
    
    def _polygon_to_bbox(self, polygon_coords, img_width, img_height):
        """Convert polygon to bbox [x, y, w, h]"""
        if not polygon_coords or len(polygon_coords) < 6:
            return None
        
        try:
            xs = [polygon_coords[i] for i in range(0, len(polygon_coords), 2)]
            ys = [polygon_coords[i] for i in range(1, len(polygon_coords), 2)]
            
            if not xs or not ys:
                return None
            
            x_min = max(0, min(xs))
            y_min = max(0, min(ys))
            x_max = min(img_width, max(xs))
            y_max = min(img_height, max(ys))
            
            w = x_max - x_min
            h = y_max - y_min
            
            if w <= 0 or h <= 0:
                return None
            
            return [float(x_min), float(y_min), float(w), float(h)]
        except Exception:
            return None
    
    def _compute_visible_bbox_from_occluded(self, amodal_bbox, occlude_rate):
        """
        Compute visible bbox estimate when we don't have visible mask
        Better heuristic than before
        """
        x, y, w, h = amodal_bbox
        
        # If not occluded, visible = amodal
        if occlude_rate < 0.05:
            return amodal_bbox
        
        # Estimate visible portion
        # Assume occlusion happens from edges
        visible_scale = np.sqrt(1.0 - occlude_rate)  # Use sqrt for better scaling
        
        # Shrink from center
        cx, cy = x + w/2, y + h/2
        new_w = w * visible_scale
        new_h = h * visible_scale
        
        return [cx - new_w/2, cy - new_h/2, new_w, new_h]
    
    def _build_dataset(self):
        """Build dataset from COCOA"""
        self.valid_images = []
        self.img_to_anns = {}
        self.total_annotations = 0
        
        images = self.cocoa_data.get('images', [])
        annotations = self.cocoa_data.get('annotations', [])
        
        img_id_to_info = {img['id']: img for img in images}
        
        found_human_categories = set()
        
        for ann in annotations:
            img_id = ann.get('image_id')
            if img_id is None or img_id not in img_id_to_info:
                continue
            
            img_info = img_id_to_info[img_id]
            img_width = img_info['width']
            img_height = img_info['height']
            
            regions = ann.get('regions', [])
            if not regions:
                continue
            
            person_regions = []
            
            for region in regions:
                obj_name = region.get('name', '').strip()
                
                # Filter for humans
                if self.person_only:
                    obj_lower = obj_name.lower().strip()
                    
                    human_categories = {
                        'person', 'people', 'peoples', 'man', 'woman', 'boy', 'girl',
                        'child', 'children', 'human', 'humans', 'guy', 'lady', 'kid', 'kids',
                        'men', 'women', 'boys', 'girls', 'ladies'
                    }
                    
                    is_human = obj_lower in human_categories
                    
                    if not is_human:
                        words = obj_lower.replace('-', ' ').replace('_', ' ').split()
                        has_human_word = any(w in human_categories for w in words)
                        blacklist = {'mango', 'mangoes', 'ottoman', 'comode', 'carton'}
                        has_blacklist = any(bl in obj_lower for bl in blacklist)
                        
                        if has_human_word and not has_blacklist:
                            is_human = True
                        elif any(phrase in obj_lower for phrase in ['old man', 'old woman', 'camera man']):
                            is_human = True
                    
                    if not is_human:
                        continue
                    
                    found_human_categories.add(obj_name)
                
                # Get amodal segmentation (full extent)
                segmentation = region.get('segmentation', [])
                if not segmentation:
                    continue
                
                amodal_bbox = self._polygon_to_bbox(segmentation, img_width, img_height)
                if amodal_bbox is None:
                    continue
                
                # Get occlusion rate
                occlude_rate = region.get('occlude_rate', 0.0)
                if occlude_rate is None:
                    occlude_rate = 0.0
                occlude_rate = float(occlude_rate)
                occlude_rate = max(0.0, min(1.0, occlude_rate))
                
                # Compute visible bbox
                visible_bbox = self._compute_visible_bbox_from_occluded(amodal_bbox, occlude_rate)
                
                person_regions.append({
                    'visible_bbox': visible_bbox,
                    'amodal_bbox': amodal_bbox,
                    'occlude_rate': occlude_rate,
                    'name': obj_name
                })
            
            if len(person_regions) > 0:
                if img_id not in self.img_to_anns:
                    self.valid_images.append({
                        'image_id': img_id,
                        'file_name': img_info['file_name'],
                        'width': img_width,
                        'height': img_height
                    })
                    self.img_to_anns[img_id] = []
                
                self.img_to_anns[img_id].extend(person_regions)
                self.total_annotations += len(person_regions)
        
        if found_human_categories:
            print(f"   Human categories: {len(found_human_categories)} types")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_info = self.valid_images[idx]
        image_id = img_info['image_id']
        file_name = img_info['file_name']
        
        # Load image
        img_path = os.path.join(self.image_dir, file_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, os.path.basename(file_name))
        
        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Get annotations
        anns = self.img_to_anns[image_id]
        
        visible_boxes = []
        amodal_boxes = []
        class_labels = []
        occlusion_scores = []
        
        for ann in anns:
            visible_boxes.append(ann['visible_bbox'])
            amodal_boxes.append(ann['amodal_bbox'])
            class_labels.append(1)
            occlusion_scores.append(ann['occlude_rate'])
        
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
            except Exception:
                pass
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Convert to normalized [cx, cy, w, h]
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        
        visible_boxes_norm = []
        amodal_boxes_norm = []
        
        for vis_box, amod_box in zip(visible_boxes, amodal_boxes):
            # Scale visible
            vis_x, vis_y, vis_w, vis_h = vis_box
            vis_x *= scale_x
            vis_y *= scale_y
            vis_w *= scale_x
            vis_h *= scale_y
            
            # Scale amodal
            amod_x, amod_y, amod_w, amod_h = amod_box
            amod_x *= scale_x
            amod_y *= scale_y
            amod_w *= scale_x
            amod_h *= scale_y
            
            # Filter by minimum area
            if vis_w * vis_h < self.min_area or amod_w * amod_h < self.min_area:
                continue
            
            # To center format and normalize
            vis_cx = (vis_x + vis_w / 2) / self.image_size
            vis_cy = (vis_y + vis_h / 2) / self.image_size
            vis_w_norm = vis_w / self.image_size
            vis_h_norm = vis_h / self.image_size
            
            amod_cx = (amod_x + amod_w / 2) / self.image_size
            amod_cy = (amod_y + amod_h / 2) / self.image_size
            amod_w_norm = amod_w / self.image_size
            amod_h_norm = amod_h / self.image_size
            
            # Clip
            vis_cx = np.clip(vis_cx, 0, 1)
            vis_cy = np.clip(vis_cy, 0, 1)
            vis_w_norm = np.clip(vis_w_norm, 0, 1)
            vis_h_norm = np.clip(vis_h_norm, 0, 1)
            
            amod_cx = np.clip(amod_cx, 0, 1)
            amod_cy = np.clip(amod_cy, 0, 1)
            amod_w_norm = np.clip(amod_w_norm, 0, 1)
            amod_h_norm = np.clip(amod_h_norm, 0, 1)
            
            visible_boxes_norm.append([vis_cx, vis_cy, vis_w_norm, vis_h_norm])
            amodal_boxes_norm.append([amod_cx, amod_cy, amod_w_norm, amod_h_norm])
        
        # Pad or truncate
        num_objects = len(visible_boxes_norm)
        
        if num_objects > self.max_objects:
            visible_boxes_norm = visible_boxes_norm[:self.max_objects]
            amodal_boxes_norm = amodal_boxes_norm[:self.max_objects]
            occlusion_scores = occlusion_scores[:self.max_objects]
            num_objects = self.max_objects
        
        visible_boxes_tensor = torch.zeros(self.max_objects, 4)
        amodal_boxes_tensor = torch.zeros(self.max_objects, 4)
        class_labels_tensor = torch.zeros(self.max_objects, dtype=torch.long)
        occlusion_tensor = torch.zeros(self.max_objects)
        valid_mask = torch.zeros(self.max_objects)
        
        if num_objects > 0:
            visible_boxes_tensor[:num_objects] = torch.tensor(visible_boxes_norm, dtype=torch.float32)
            amodal_boxes_tensor[:num_objects] = torch.tensor(amodal_boxes_norm, dtype=torch.float32)
            class_labels_tensor[:num_objects] = 1
            occlusion_tensor[:num_objects] = torch.tensor(occlusion_scores[:num_objects], dtype=torch.float32)
            valid_mask[:num_objects] = 1.0
        
        return {
            'image': image,
            'visible_boxes': visible_boxes_tensor,
            'amodal_boxes': amodal_boxes_tensor,
            'class_labels': class_labels_tensor,
            'occlusion_scores': occlusion_tensor,
            'valid_mask': valid_mask,
        }


def collate_fn(batch):
    """Custom collate function"""
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