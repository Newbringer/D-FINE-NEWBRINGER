#!/usr/bin/env python3
"""
COCOA Dataset for HUMANS ONLY
Simple, clean, no BS - just human amodal detection
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
    COCOA Dataset for HUMANS ONLY
    - Uses Official COCOA (has human categories like man, woman, boy, girl)
    - Falls back to Detectron COCOA for images not in Official
    - Filters STRICTLY for humans (no mangoes, snowmen, etc.)
    - Optional: Train ONLY on occluded samples (min_occlusion > 0)
    """
    
    def __init__(self,
                 image_dir: str,
                 official_ann_file: str,
                 detectron_ann_file: str = None,
                 split: str = 'train',
                 image_size: int = 640,
                 max_objects: int = 50,
                 augment: bool = True,
                 min_occlusion: float = 0.0,
                 min_area: int = 400):
        """
        Args:
            official_ann_file: Official COCOA (has human categories)
            detectron_ann_file: Detectron COCOA (backup for extra data)
            min_occlusion: 0.0=all samples, 0.05=slightly occluded+, 0.1=clearly occluded
        """
        self.image_dir = image_dir
        self.split = split
        self.image_size = image_size
        self.max_objects = max_objects
        self.min_occlusion = min_occlusion
        self.min_area = min_area
        self.use_detectron = detectron_ann_file is not None

        print(f"📂 Loading COCOA {split} - HUMANS ONLY")
        print(f"   Official: {official_ann_file}")
        if self.use_detectron:
            print(f"   Detectron backup: {detectron_ann_file}")
        print(f"   Min occlusion: {min_occlusion:.2f}")
        
        # Load Official COCOA
        with open(official_ann_file, 'r') as f:
            self.official_data = json.load(f)
        
        # Load Detectron COCOA if provided
        self.detectron_data = None
        if self.use_detectron:
            with open(detectron_ann_file, 'r') as f:
                self.detectron_data = json.load(f)
        
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
        
        print(f"✅ Dataset loaded:")
        print(f"   Images: {len(self.valid_images)}")
        if self.use_detectron:
            print(f"   From Official: {self.from_official}")
            print(f"   From Detectron: {self.from_detectron}")
        print(f"   Human annotations: {self.total_annotations}")
        if len(self.valid_images) > 0:
            print(f"   Avg per image: {self.total_annotations / len(self.valid_images):.2f}")
    
    def _is_human(self, name: str) -> bool:
        """STRICT human filtering - no mangoes allowed!"""
        name_lower = name.lower().strip()
        
        # Exact matches ONLY
        human_exact = {
            'person', 'people', 'peoples', 'man', 'woman', 'boy', 'girl',
            'child', 'children', 'human', 'humans', 'guy', 'lady', 'kid', 'kids',
            'men', 'women', 'boys', 'girls', 'ladies'
        }
        
        # Blacklist - DO NOT include these
        blacklist = {
            'mango', 'mangoes', 'ottoman', 'snowman', 'snowmen',
            'statue', 'carton', 'comode'
        }
        
        # Check blacklist first
        if any(bad in name_lower for bad in blacklist):
            return False
        
        # Exact match
        if name_lower in human_exact:
            return True
        
        # Word-level match (for "old man", "young boy", etc.)
        words = name_lower.replace('-', ' ').replace('_', ' ').split()
        if any(w in human_exact for w in words):
            return True
        
        return False
    
    def _polygon_to_bbox(self, polygon_coords):
        """Convert polygon to [x, y, w, h]"""
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
    
    def _estimate_visible_bbox(self, amodal_bbox, occlude_rate):
        """Estimate visible bbox from amodal + occlusion rate"""
        x, y, w, h = amodal_bbox
        
        # No occlusion = visible same as amodal
        if occlude_rate < 0.05:
            return amodal_bbox
        
        # Estimate visible portion (shrink from center)
        scale = np.sqrt(1.0 - occlude_rate)
        cx, cy = x + w/2, y + h/2
        new_w = w * scale
        new_h = h * scale
        
        return [cx - new_w/2, cy - new_h/2, new_w, new_h]
    
    def _build_dataset(self):
        """Build dataset from COCOA - HUMANS ONLY, with Detectron backup"""
        self.valid_images = []
        self.img_to_anns = {}
        self.total_annotations = 0
        self.from_official = 0
        self.from_detectron = 0
        
        # Get images from Official
        official_images = self.official_data.get('images', [])
        official_annotations = self.official_data.get('annotations', [])
        
        img_id_to_info = {img['id']: img for img in official_images}
        
        # Process Official COCOA - HUMANS ONLY
        official_img_ids_with_humans = set()
        
        for ann in official_annotations:
            img_id = ann.get('image_id')
            if img_id is None or img_id not in img_id_to_info:
                continue
            
            img_info = img_id_to_info[img_id]
            
            regions = ann.get('regions', [])
            if not regions:
                continue
            
            human_regions = []
            
            for region in regions:
                obj_name = region.get('name', '').strip()
                
                # STRICT human filtering
                if not self._is_human(obj_name):
                    continue
                
                # Get amodal bbox from segmentation
                seg = region.get('segmentation', [])
                amodal_bbox = self._polygon_to_bbox(seg)
                if amodal_bbox is None:
                    continue
                
                # Get occlusion
                occ = region.get('occlude_rate', 0)
                if occ is None:
                    occ = 0
                occ = float(occ)
                occ = max(0.0, min(1.0, occ))
                
                # Filter by minimum occlusion
                if occ < self.min_occlusion:
                    continue
                
                # Estimate visible bbox
                visible_bbox = self._estimate_visible_bbox(amodal_bbox, occ)
                
                human_regions.append({
                    'visible_bbox': visible_bbox,
                    'amodal_bbox': amodal_bbox,
                    'occlude_rate': occ,
                    'name': obj_name,
                    'source': 'official'
                })
            
            if len(human_regions) > 0:
                official_img_ids_with_humans.add(img_id)
                
                if img_id not in self.img_to_anns:
                    self.valid_images.append({
                        'image_id': img_id,
                        'file_name': img_info['file_name'],
                        'width': img_info['width'],
                        'height': img_info['height'],
                        'source': 'official'
                    })
                    self.img_to_anns[img_id] = []
                    self.from_official += 1
                
                self.img_to_anns[img_id].extend(human_regions)
                self.total_annotations += len(human_regions)
        
        # Process Detectron COCOA as backup (for images not in Official or without humans)
        if self.use_detectron and self.detectron_data:
            detectron_images = {img['id']: img for img in self.detectron_data.get('images', [])}
            detectron_annotations = self.detectron_data.get('annotations', [])
            
            # Group detectron annotations by image (FILTER FOR PERSON ONLY!)
            detectron_by_img = {}
            for ann in detectron_annotations:
                # FILTER: Only use category_id == 1 (person in COCO)
                if ann.get('category_id') != 1:
                    continue
                    
                img_id = ann.get('image_id')
                if img_id not in detectron_by_img:
                    detectron_by_img[img_id] = []
                detectron_by_img[img_id].append(ann)
            
            # Add detectron data for images NOT in official humans
            for img_id, anns in detectron_by_img.items():
                # Skip if Official already has humans for this image
                if img_id in official_img_ids_with_humans:
                    continue
                
                if img_id not in detectron_images:
                    continue
                
                img_info = detectron_images[img_id]
                
                detectron_regions = []
                
                for ann in anns:
                    # Get occlusion
                    occ = ann.get('occlude_rate', 0)
                    if occ is None:
                        occ = 0
                    occ = float(occ)
                    occ = max(0.0, min(1.0, occ))
                    
                    # Filter by minimum occlusion
                    if occ < self.min_occlusion:
                        continue
                    
                    # Get amodal bbox (pre-computed)
                    amodal_bbox = ann.get('bbox')
                    if not amodal_bbox or len(amodal_bbox) != 4:
                        continue
                    
                    amodal_bbox = tuple(amodal_bbox)
                    
                    # Try to get visible bbox from visible_mask
                    visible_mask = ann.get('visible_mask')
                    if visible_mask and len(visible_mask) >= 6:
                        visible_bbox = self._polygon_to_bbox(visible_mask)
                        if visible_bbox is None:
                            visible_bbox = self._estimate_visible_bbox(amodal_bbox, occ)
                    else:
                        visible_bbox = self._estimate_visible_bbox(amodal_bbox, occ)
                    
                    detectron_regions.append({
                        'visible_bbox': visible_bbox,
                        'amodal_bbox': amodal_bbox,
                        'occlude_rate': occ,
                        'name': 'person',
                        'source': 'detectron'
                    })
                
                if len(detectron_regions) > 0:
                    if img_id not in self.img_to_anns:
                        self.valid_images.append({
                            'image_id': img_id,
                            'file_name': img_info['file_name'],
                            'width': img_info['width'],
                            'height': img_info['height'],
                            'source': 'detectron'
                        })
                        self.img_to_anns[img_id] = []
                        self.from_detectron += 1
                    
                    self.img_to_anns[img_id].extend(detectron_regions)
                    self.total_annotations += len(detectron_regions)
    
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
            except:
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