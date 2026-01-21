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
    
    def __init__(self, root_dir, split='train', class_id=80, image_size=640):
        """
        Args:
            root_dir: Root directory with images and _annotations.coco.json
            split: 'train' or 'val' (will auto-split if only one json)
            class_id: Class ID for glass wall (default: 80)
            image_size: Target image size (default: 640)
        """
        self.root_dir = Path(root_dir)
        self.class_id = class_id
        self.image_size = image_size
        self.split = split
        
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
        
        print(f"   Loaded {len(self.images)} {split} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = self.root_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        
        # Convert annotations
        boxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            
            # Scale to resized image
            x = x * self.image_size / orig_w
            y = y * self.image_size / orig_h
            w = w * self.image_size / orig_w
            h = h * self.image_size / orig_h
            
            # Convert to xyxy format
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_id)  # All glass annotations get class 80
        
        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
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
        
        # Load COCO annotations
        ann_file = self.root_dir / 'annotations' / 'instances_train2017.json'
        if not ann_file.exists():
            # Try alternative path
            ann_file = self.root_dir / 'instances_train2017.json'
        
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotations not found: {ann_file}")
        
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
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
        
        # Image directory
        self.img_dir = self.root_dir / 'images' / 'train2017'
        if not self.img_dir.exists():
            self.img_dir = self.root_dir / 'train2017'
        
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
            
            # Scale to resized image
            x = x * self.image_size / orig_w
            y = y * self.image_size / orig_h
            w = w * self.image_size / orig_w
            h = h * self.image_size / orig_h
            
            # Convert to xyxy
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            
            # COCO category IDs are 1-90, need to map to 0-79
            # Simplified: just use category_id - 1 (won't be perfect but close enough)
            label = min(category_id - 1, 79)  # Ensure it's in 0-79 range
            
            boxes.append([x1, y1, x2, y2])
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