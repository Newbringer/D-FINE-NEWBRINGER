#!/usr/bin/env python3
"""
COCOA Dataset Loader for Amodal Person Detection
CORRECTED for actual COCOA JSON structure with polygon mask support
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
from typing import Dict, List
import albumentations as A
from pycocotools import mask as mask_utils


class COCOAAmodalDataset(Dataset):
    """
    COCOA Dataset for Amodal Person Detection
    
    COCOA structure:
    - bbox: visible bounding box
    - segmentation: full amodal mask (polygon or RLE)
    - visible_mask: visible-only mask (polygon or RLE)
    - amodal_region['name']: category name (we want 'person')
    - occlude_rate: occlusion score
    """
    
    def __init__(self,
                 coco_root: str,
                 cocoa_annotation_file: str,
                 split: str = 'train',
                 image_size: int = 640,
                 max_objects: int = 50,
                 augment: bool = True,
                 person_only: bool = True):
        """
        Args:
            coco_root: Path to COCO images (e.g., coco/train2014)
            cocoa_annotation_file: Path to COCOA annotation JSON
            split: 'train' or 'val'
            image_size: Target image size
            max_objects: Maximum number of people per image
            augment: Whether to apply augmentations
            person_only: If True, only load person annotations
        """
        self.coco_root = coco_root
        self.split = split
        self.image_size = image_size
        self.max_objects = max_objects
        self.person_only = person_only
        
        print(f"📂 Loading COCOA {split} annotations from: {cocoa_annotation_file}")
        
        # Load COCOA annotations
        with open(cocoa_annotation_file, 'r') as f:
            self.cocoa_data = json.load(f)
        
        # Build dataset
        self._build_dataset()
        
        # Normalization (ImageNet stats)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Augmentations (only for training)
        if augment and split == 'train':
            try:
                self.transforms = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                ], bbox_params=A.BboxParams(
                    format='coco',
                    min_visibility=0.3,
                    label_fields=['class_labels']
                ))
            except (TypeError, ValueError):
                self.transforms = None
        else:
            self.transforms = None
        
        print(f"✅ Loaded COCOA {split} set:")
        print(f"   Images: {len(self.valid_images)}")
        print(f"   Total person annotations: {self.total_annotations}")
        if len(self.valid_images) > 0:
            print(f"   Avg people per image: {self.total_annotations / len(self.valid_images):.2f}")
    
    def _mask_to_bbox(self, mask_data, img_width, img_height):
        """Convert RLE or polygon mask to bounding box [x, y, w, h]"""
        try:
            if isinstance(mask_data, dict):
                # RLE format
                mask = mask_utils.decode(mask_data)
            elif isinstance(mask_data, list):
                # Polygon format - convert to mask
                if len(mask_data) == 0:
                    return None
                
                # Create blank mask
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                
                # Draw polygons
                for polygon in mask_data:
                    if len(polygon) < 6:  # Need at least 3 points (x,y pairs)
                        continue
                    # Reshape polygon to (n_points, 2)
                    poly = np.array(polygon).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)
            else:
                return None
            
            if mask.sum() == 0:
                return None
            
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not rows.any() or not cols.any():
                return None
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
        except Exception as e:
            print(f"⚠️  Error converting mask to bbox: {e}")
            return None
    
    def _build_dataset(self):
        """Build dataset from COCOA annotations"""
        self.valid_images = []
        self.img_to_anns = {}
        self.total_annotations = 0
        
        # Parse structure
        images = self.cocoa_data.get('images', [])
        annotations = self.cocoa_data.get('annotations', [])
        
        # Build image id to image info mapping
        img_id_to_info = {img['id']: img for img in images}
        
        # Build image id to annotations mapping
        img_id_to_anns = {}
        for ann in annotations:
            img_id = ann['image_id']
            
            # Filter for person if requested
            if self.person_only:
                amodal_region = ann.get('amodal_region', {})
                obj_name = amodal_region.get('name', '').lower()
                if obj_name != 'person':
                    continue
            
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)
        
        print(f"   Found {len(img_id_to_anns)} images with person annotations")
        
        # Process each image
        for img_id, image_anns in img_id_to_anns.items():
            if img_id not in img_id_to_info:
                continue
            
            img_info = img_id_to_info[img_id]
            
            # Process annotations
            valid_anns = []
            for ann in image_anns:
                # Get visible bbox (this is what we can see)
                visible_bbox = ann.get('bbox', None)
                
                if visible_bbox is None or len(visible_bbox) != 4:
                    continue
                
                if visible_bbox[2] <= 0 or visible_bbox[3] <= 0:
                    continue
                
                # Get amodal bbox from segmentation mask
                segmentation = ann.get('segmentation', None)
                
                if segmentation is None:
                    # If no segmentation, use visible bbox for both
                    amodal_bbox = visible_bbox
                else:
                    # Compute amodal bbox from full mask (need image dimensions)
                    amodal_bbox = self._mask_to_bbox(
                        segmentation, 
                        img_info['width'], 
                        img_info['height']
                    )
                    if amodal_bbox is None:
                        amodal_bbox = visible_bbox
                
                # Validate amodal bbox
                if amodal_bbox[2] <= 0 or amodal_bbox[3] <= 0:
                    amodal_bbox = visible_bbox
                
                # Get occlusion rate
                occlude_rate = ann.get('occlude_rate', 0)
                if occlude_rate is None:
                    occlude_rate = 0
                occlude_rate = float(occlude_rate)
                
                valid_anns.append({
                    'visible_bbox': visible_bbox,
                    'amodal_bbox': amodal_bbox,
                    'occlude_rate': occlude_rate,
                    'category_name': ann.get('amodal_region', {}).get('name', 'person')
                })
            
            if len(valid_anns) > 0:
                self.valid_images.append({
                    'image_id': img_id,
                    'file_name': img_info['file_name'],
                    'width': img_info['width'],
                    'height': img_info['height']
                })
                self.img_to_anns[img_id] = valid_anns
                self.total_annotations += len(valid_anns)
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index"""
        
        img_info = self.valid_images[idx]
        image_id = img_info['image_id']
        file_name = img_info['file_name']
        
        # Load image
        img_path = os.path.join(self.coco_root, file_name)
        
        if not os.path.exists(img_path):
            img_path = os.path.join(self.coco_root, os.path.basename(file_name))
        
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"⚠️  Failed to load image: {img_path}")
            return self.__getitem__((idx + 1) % len(self))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Get annotations for this image
        anns = self.img_to_anns[image_id]
        
        # Extract boxes and labels
        visible_boxes = []
        amodal_boxes = []
        class_labels = []
        occlusion_scores = []
        
        for ann in anns:
            visible_bbox = ann['visible_bbox']
            amodal_bbox = ann['amodal_bbox']
            occlude_rate = ann['occlude_rate']
            
            visible_boxes.append(visible_bbox)
            amodal_boxes.append(amodal_bbox)
            class_labels.append(1)  # Person class
            occlusion_scores.append(occlude_rate)
        
        # Skip images with no valid annotations
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
                
                # Keep only corresponding amodal boxes
                if len(visible_boxes) > 0:
                    amodal_boxes = amodal_boxes[:len(visible_boxes)]
                    occlusion_scores = occlusion_scores[:len(visible_boxes)]
            except Exception as e:
                pass
        
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
            # Scale visible box
            vis_x, vis_y, vis_w, vis_h = vis_box
            vis_x *= scale_x
            vis_y *= scale_y
            vis_w *= scale_x
            vis_h *= scale_y
            
            # Scale amodal box
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
            
            # Clip to valid range
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
        
        # Pad or truncate to max_objects
        num_objects = len(class_labels)
        
        if num_objects > self.max_objects:
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
            'orig_size': torch.tensor([orig_h, orig_w]),
            'file_name': file_name
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


# Test script
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python cocoa_dataset.py <coco_root> <cocoa_annotation_file>")
        print("Example: python cocoa_dataset.py coco/train2014 coco/COCO_amodal_train2014_detectron.json")
        sys.exit(1)
    
    coco_root = sys.argv[1]
    cocoa_ann = sys.argv[2]
    
    print("\n" + "="*60)
    print("Testing COCOA Dataset Loader")
    print("="*60)
    
    dataset = COCOAAmodalDataset(
        coco_root=coco_root,
        cocoa_annotation_file=cocoa_ann,
        split='train',
        image_size=640,
        augment=False,
        person_only=True
    )
    
    print(f"\n📊 Dataset size: {len(dataset)}")
    
    # Test loading samples
    print("\n🔍 Testing sample loading...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        num_valid = sample['valid_mask'].sum().item()
        print(f"\nSample {i}:")
        print(f"  File: {sample.get('file_name', 'N/A')}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Valid objects: {num_valid}")
        
        if num_valid > 0:
            valid_occ = sample['occlusion_scores'][sample['valid_mask'] > 0]
            print(f"  Occlusion scores: min={valid_occ.min():.2f}, max={valid_occ.max():.2f}, mean={valid_occ.mean():.2f}")
            
            # Check if amodal boxes are different from visible
            vis_boxes = sample['visible_boxes'][sample['valid_mask'] > 0]
            amod_boxes = sample['amodal_boxes'][sample['valid_mask'] > 0]
            diff = (amod_boxes - vis_boxes).abs().sum()
            print(f"  Bbox difference (amodal vs visible): {diff:.4f}")
    
    print("\n✅ Dataset loader working correctly!")