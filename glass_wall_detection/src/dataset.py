#!/usr/bin/env python3
"""
Dataset classes for glass wall detection training
"""

import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import random


class GlassWallDataset(Dataset):
    """Glass wall detection dataset"""
    
    def __init__(self, root_dir, split='train', class_id=80, image_size=640,
                 augment=True, augment_multiplier=1):
        """
        Args:
            root_dir: Root directory with images and _annotations.coco.json
            split: 'train' or 'val' (will auto-split if only one json)
            class_id: Class ID for glass wall (default: 80)
            image_size: Target image size (default: 640)
            augment: Apply aggressive augmentation (default: True for train)
        """
        self.root_dir = Path(root_dir)
        self.class_id = class_id
        self.image_size = image_size
        self.split = split
        self.augment = augment and (split in ['train', 'all'])  # Only augment training data
        self.augment_multiplier = max(1, int(augment_multiplier)) if self.augment else 1
        
        # Load COCO annotations
        json_path = self.root_dir / '_annotations.coco.json'
        if not json_path.exists():
            raise FileNotFoundError(f"Annotations not found: {json_path}")
        
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Split into train/val if needed
        all_images = self.coco_data['images']
        
        # If split requested, do 80/20 train/val split
        if split == 'train':
            self.images = all_images[:int(0.8 * len(all_images))]
        elif split == 'val':
            self.images = all_images[int(0.8 * len(all_images)):]
        else:
            self.images = all_images
        
        # Create image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        uses_multiplier = self.split in ['train', 'all']
        total = len(self.images) * self.augment_multiplier if uses_multiplier else len(self.images)
        if uses_multiplier and self.augment_multiplier > 1:
            print(f"   Loaded {len(self.images)} {split} images (effective {total}, x{self.augment_multiplier} aug)")
        else:
            print(f"   Loaded {len(self.images)} {split} images (effective {total})")
    
    def __len__(self):
        if self.split in ['train', 'all']:
            return len(self.images) * self.augment_multiplier
        return len(self.images)
    
    def __getitem__(self, idx):
        base_len = len(self.images)
        base_idx = idx % base_len
        force_augment = self.augment and idx >= base_len
        img_info = self.images[base_idx]
        img_id = img_info['id']
        
        # Load image
        img_path = self.root_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        
        # Convert annotations to absolute xyxy boxes
        boxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            
            x1 = max(0.0, min(float(x), orig_w - 1.0))
            y1 = max(0.0, min(float(y), orig_h - 1.0))
            x2 = max(0.0, min(float(x + w), orig_w - 1.0))
            y2 = max(0.0, min(float(y + h), orig_h - 1.0))
            
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_id)  # All glass annotations get class 80
        
        # Apply augmentation BEFORE resize for better quality
        if self.augment or force_augment:
            image, boxes, labels = self._apply_augmentation(image, boxes, labels)
        
        # Use augmented image size for normalization
        aug_h, aug_w = image.shape[:2]
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert to normalized cxcywh after augmentation
        
        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5 / aug_w
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5 / aug_h
            w_norm = (boxes[:, 2] - boxes[:, 0]) / aug_w
            h_norm = (boxes[:, 3] - boxes[:, 1]) / aug_h
            boxes = np.stack([cx, cy, w_norm, h_norm], axis=1)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = []
        
        # Convert to tensors
        has_boxes = len(boxes) > 0
        boxes = torch.tensor(boxes, dtype=torch.float32) if has_boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64)
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor (CHW format)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([orig_h, orig_w])
        }
        
        return image, target
    
    def _apply_augmentation(self, image, boxes, labels):
        """Apply LIGHT augmentation for specific marked glass panes
        
        Use case: 4 specific glass panes with "01", "02", "03", "04" markers
        Goal: Slight variations in lighting/angle while keeping markers recognizable
        
        We want the model to OVERFIT to these specific panes!
        """
        import random
        
        # NO horizontal flip - would swap pane positions (01 becomes mirrored)
        
        # Slight brightness adjustment (±20%) - office lighting changes
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        
        # Slight contrast adjustment (±20%) - camera exposure
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = image.mean(axis=(0, 1), keepdims=True)
            image = np.clip((image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        # Color jitter in HSV space (small shifts)
        if random.random() > 0.6:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            h_shift = random.uniform(-3, 3)
            s_scale = random.uniform(0.85, 1.2)
            v_scale = random.uniform(0.85, 1.2)
            hsv[..., 0] = (hsv[..., 0] + h_shift) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * s_scale, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * v_scale, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Slight Gaussian blur (camera focus, 25% chance)
        if random.random() > 0.75:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Mild scale/translate to mimic slight camera shifts
        if boxes:
            h, w = image.shape[:2]
            if random.random() > 0.6:
                scale = random.uniform(0.95, 1.05)
                tx = random.uniform(-0.05, 0.05) * w
                ty = random.uniform(-0.05, 0.05) * h
                
                M = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
                image = cv2.warpAffine(
                    image,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                
                updated_boxes = []
                updated_labels = []
                for (x1, y1, x2, y2), label in zip(boxes, labels):
                    x1 = scale * x1 + tx
                    x2 = scale * x2 + tx
                    y1 = scale * y1 + ty
                    y2 = scale * y2 + ty
                    
                    x1 = max(0.0, min(x1, w - 1.0))
                    x2 = max(0.0, min(x2, w - 1.0))
                    y1 = max(0.0, min(y1, h - 1.0))
                    y2 = max(0.0, min(y2, h - 1.0))
                    
                    if x2 > x1 and y2 > y1:
                        updated_boxes.append([x1, y1, x2, y2])
                        updated_labels.append(label)
                
                boxes = updated_boxes
                labels = updated_labels
        
        # Mild rotation (±5 degrees) to mimic viewpoint change
        if boxes:
            h, w = image.shape[:2]
            if random.random() > 0.65:
                angle = random.uniform(-5.0, 5.0)
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                image = cv2.warpAffine(
                    image,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                
                updated_boxes = []
                updated_labels = []
                for (x1, y1, x2, y2), label in zip(boxes, labels):
                    corners = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.float32
                    )
                    ones = np.ones((4, 1), dtype=np.float32)
                    corners_h = np.hstack([corners, ones])
                    rotated = corners_h @ M.T
                    rx1 = float(np.min(rotated[:, 0]))
                    ry1 = float(np.min(rotated[:, 1]))
                    rx2 = float(np.max(rotated[:, 0]))
                    ry2 = float(np.max(rotated[:, 1]))
                    
                    rx1 = max(0.0, min(rx1, w - 1.0))
                    rx2 = max(0.0, min(rx2, w - 1.0))
                    ry1 = max(0.0, min(ry1, h - 1.0))
                    ry2 = max(0.0, min(ry2, h - 1.0))
                    
                    if rx2 > rx1 and ry2 > ry1:
                        updated_boxes.append([rx1, ry1, rx2, ry2])
                        updated_labels.append(label)
                
                boxes = updated_boxes
                labels = updated_labels
        
        return image, boxes, labels


class COCORetentionDataset(Dataset):
    """Small COCO subset for preventing forgetting"""
    
    def __init__(self, root_dir, max_images=200, image_size=640):
        """
        Args:
            root_dir: COCO dataset root (should have images/train2017/ and annotations/)
            max_images: Maximum number of images to use
            image_size: Target image size
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.max_images = max_images
        
        # Load COCO annotations (auto-select train/val based on available images)
        ann_candidates = [
            self.root_dir / 'annotations' / 'instances_train2017.json',
            self.root_dir / 'annotations' / 'instances_val2017.json',
            self.root_dir / 'instances_train2017.json',
            self.root_dir / 'instances_val2017.json',
        ]

        best = {"hits": -1, "ann_file": None, "data": None, "img_dir": None}
        for ann_file in ann_candidates:
            if not ann_file.exists():
                continue
            with open(ann_file, 'r') as f:
                data = json.load(f)

            img_dir = self._resolve_image_dir_for_data(self.root_dir, data)
            # Score by matching first N filenames
            sample_names = [img.get('file_name') for img in data.get('images', [])[:200] if img.get('file_name')]
            hits = 0
            for name in sample_names:
                if (img_dir / name).exists():
                    hits += 1
            if hits > best["hits"]:
                best = {"hits": hits, "ann_file": ann_file, "data": data, "img_dir": img_dir}

        if best["ann_file"] is None:
            raise FileNotFoundError("COCO annotations not found in expected locations")
        if best["hits"] == 0:
            raise FileNotFoundError(
                f"COCO images not found for annotations: {best['ann_file']}. "
                f"Resolved image dir: {best['img_dir']}"
            )

        self.coco_data = best["data"]
        self.img_dir = best["img_dir"]
        
        # Sample random images
        all_images = self.coco_data['images']
        random.shuffle(all_images)
        self.images = all_images[:max_images]
        
        # Create mappings
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        print(f"   COCO annotations: {best['ann_file'].name}")
        print(f"   COCO images dir: {self.img_dir}")
        
        print(f"   Loaded {len(self.images)} COCO images for retention")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = self.img_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        if image is None:
            # Return empty sample if image not found
            return self._get_empty_sample()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        
        for ann in anns:
            if 'bbox' not in ann:
                continue
                
            x, y, w, h = ann['bbox']
            category_id = ann['category_id']
            
            # Convert COCO xywh (pixels) -> normalized cxcywh (0-1)
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            w_norm = w / orig_w
            h_norm = h / orig_h
            
            # COCO category IDs are 1-90, need to map to 0-79
            # Simplified: just use category_id - 1 (won't be perfect but close enough)
            label = min(category_id - 1, 79)  # Ensure it's in 0-79 range
            
            boxes.append([cx, cy, w_norm, h_norm])
            labels.append(label)
        
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([orig_h, orig_w])
        }
        
        return image, target
    
    def _get_empty_sample(self):
        """Return empty sample if image loading fails"""
        image = torch.zeros(3, self.image_size, self.image_size)
        target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([0]),
            'orig_size': torch.tensor([self.image_size, self.image_size])
        }
        return image, target

    @staticmethod
    def _resolve_image_dir_for_data(root_dir: Path, coco_data: dict):
        """Resolve COCO image directory based on annotation filenames."""
        # Common COCO layouts (including nested val2017 under train2017)
        candidates = [
            root_dir / 'images' / 'train2017' / 'val2017',
            root_dir / 'images' / 'train2017',
            root_dir / 'images' / 'val2017',
            root_dir / 'train2017',
            root_dir / 'val2017',
            root_dir / 'images',
        ]

        # Try to validate against the first image filename
        sample_name = None
        if coco_data.get('images'):
            sample_name = coco_data['images'][0].get('file_name')

        if sample_name:
            # If file_name already contains subdirs, try root/images/<file_name> first
            direct_candidates = [
                root_dir / sample_name,
                root_dir / 'images' / sample_name,
            ]
            for candidate in direct_candidates:
                if candidate.exists():
                    return candidate.parent

            for candidate in candidates:
                if (candidate / sample_name).exists():
                    return candidate

        # Fallback to first existing directory
        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Last resort: root_dir
        return self.root_dir


class MixedDataset(Dataset):
    """Mix glass and COCO datasets"""
    
    def __init__(self, glass_dataset, coco_dataset, glass_ratio=0.7):
        """
        Args:
            glass_dataset: GlassWallDataset
            coco_dataset: COCORetentionDataset
            glass_ratio: Ratio of glass images (0.7 = 70% glass, 30% COCO)
        """
        self.glass_dataset = glass_dataset
        self.coco_dataset = coco_dataset
        self.glass_ratio = glass_ratio
        
        # Calculate lengths
        glass_len = len(glass_dataset)
        coco_len = int(glass_len * (1 - glass_ratio) / glass_ratio)
        
        self.total_length = glass_len + coco_len
        
        print(f"   Mixed dataset: {glass_len} glass + {coco_len} COCO = {self.total_length} total")
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Randomly choose glass or COCO
        if random.random() < self.glass_ratio:
            # Glass sample
            idx = random.randint(0, len(self.glass_dataset) - 1)
            return self.glass_dataset[idx]
        else:
            # COCO sample
            idx = random.randint(0, len(self.coco_dataset) - 1)
            return self.coco_dataset[idx]


def collate_fn(batch):
    """Custom collate function for variable-size targets"""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets