#!/usr/bin/env python3
"""
Create Synthetic Amodal Dataset from COCO
Generates controlled occlusions: tables, walls, furniture, out-of-frame

Target: 4,000-6,000 high-quality instances
Occlusion types:
- Bottom: 30% (tables, desks)
- Side: 30% (walls, doorframes)  
- Corner: 20% (furniture)
- Out-of-frame: 20% (partial body)
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random
from tqdm import tqdm
from pycocotools.coco import COCO


class SyntheticAmodalGenerator:
    """Generate synthetic amodal annotations from COCO"""
    
    def __init__(self, 
                 coco_images_dir: str,
                 coco_ann_file: str,
                 output_dir: str,
                 target_samples: int = 5000,
                 bottom_ratio: float = 0.40,
                 side_ratio: float = 0.30,
                 corner_ratio: float = 0.20,
                 out_of_frame_ratio: float = 0.10,
                 no_occlusion_ratio: float = 0.20):
        """
        Args:
            no_occlusion_ratio: Ratio of non-occluded samples (important!)
        """
        self.coco_images_dir = coco_images_dir
        self.output_dir = output_dir
        self.target_samples = target_samples
        self.no_occ_ratio = no_occlusion_ratio
        
        # Balanced distribution for stable learning
        # 10% minimal (0-15%) - baseline + easy cases
        # 35% low-medium (15-45%) - common cases
        # 35% medium-high (45-75%) - challenging
        # 20% very high (75-90%) - extreme
        self.occ_ranges = [
            (0.00, 0.15, 0.10),
            (0.15, 0.45, 0.35),
            (0.45, 0.75, 0.35),
            (0.75, 0.90, 0.20)
        ]
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"🔧 SYNTHETIC AMODAL DATASET GENERATOR")
        print(f"{'='*80}")
        print(f"COCO images: {coco_images_dir}")
        print(f"COCO annotations: {coco_ann_file}")
        print(f"Output: {output_dir}")
        print(f"Target samples: {target_samples}")
        print(f"{'='*80}\n")
        
        # Load COCO
        print("📂 Loading COCO annotations...")
        self.coco = COCO(coco_ann_file)
        
        # Get person category
        self.person_cat_id = self.coco.getCatIds(catNms=['person'])[0]
        
        # Get all person annotations
        self.person_ann_ids = self.coco.getAnnIds(catIds=[self.person_cat_id])
        self.person_anns = self.coco.loadAnns(self.person_ann_ids)
        
        # Filter for good annotations
        self.person_anns = [
            ann for ann in self.person_anns
            if ann['area'] > 2000 and  # Reasonable size
               ann['bbox'][2] > 30 and ann['bbox'][3] > 50  # Min width/height
        ]
        
        print(f"✅ Found {len(self.person_anns)} person annotations")
        print(f"   Will generate {target_samples} synthetic instances\n")
        
        # Occlusion distribution (configurable)
        side_left_ratio = side_ratio / 2
        side_right_ratio = side_ratio / 2
        
        self.occlusion_types = [
            ('bottom', bottom_ratio),
            ('side_left', side_left_ratio),
            ('side_right', side_right_ratio),
            ('corner', corner_ratio),
            ('out_of_frame', out_of_frame_ratio)
        ]
    
    def _get_occlusion_mask(self, 
                           img_shape: Tuple[int, int],
                           bbox: List[float],
                           occ_type: str) -> Tuple[np.ndarray, List[float]]:
        """
        Create occlusion mask with progressive difficulty
        """
        h, w = img_shape[:2]
        x, y, bw, bh = bbox
        
        mask = np.ones((h, w), dtype=np.uint8)
        
        # Sample occlusion ratio from progressive distribution
        ranges_flat = [(r[0], r[1], r[2]) for r in self.occ_ranges]
        range_choice = random.choices(
            range(len(ranges_flat)),
            weights=[r[2] for r in ranges_flat]
        )[0]
        
        min_occ, max_occ, _ = ranges_flat[range_choice]
        occ_ratio = random.uniform(min_occ, max_occ)
        
        # No occlusion case
        if occ_ratio < 0.05:
            return mask, list(bbox)
        
        if occ_type == 'bottom':
            # ONLY expand bottom
            occ_height = int(bh * occ_ratio)
            y_cut = int(y + bh - occ_height)
            mask[y_cut:, :] = 0
            visible_bbox = [x, y, bw, max(1, bh - occ_height)]
        
        elif occ_type == 'side_left':
            # ONLY expand left
            occ_width = int(bw * occ_ratio)
            mask[:, :int(x + occ_width)] = 0
            visible_bbox = [x + occ_width, y, max(1, bw - occ_width), bh]
        
        elif occ_type == 'side_right':
            # ONLY expand right
            occ_width = int(bw * occ_ratio)
            x_cut = int(x + bw - occ_width)
            mask[:, x_cut:] = 0
            visible_bbox = [x, y, max(1, bw - occ_width), bh]
        
        elif occ_type == 'corner':
            corner = random.choice(['bottom_left', 'bottom_right'])
            
            if corner == 'bottom_left':
                occ_h = int(bh * occ_ratio)
                occ_w = int(bw * occ_ratio)
                mask[int(y + bh - occ_h):, :int(x + occ_w)] = 0
                visible_bbox = [x + occ_w*0.3, y, bw - occ_w*0.3, bh - occ_h*0.3]
            else:  # bottom_right
                occ_h = int(bh * occ_ratio)
                occ_w = int(bw * occ_ratio)
                mask[int(y + bh - occ_h):, int(x + bw - occ_w):] = 0
                visible_bbox = [x, y, bw - occ_w*0.3, bh - occ_h*0.3]
        
        elif occ_type == 'out_of_frame':
            edge = random.choice(['bottom', 'left', 'right'])
            
            if edge == 'bottom':
                crop_h = int(h * occ_ratio * 0.6)
                mask[h - crop_h:, :] = 0
                visible_bbox = [x, y, bw, max(1, min(bh, h - crop_h - y))]
            elif edge == 'left':
                crop_w = int(w * occ_ratio * 0.6)
                mask[:, :crop_w] = 0
                visible_bbox = [max(x, crop_w), y, max(1, bw - max(0, crop_w - x)), bh]
            else:  # right
                crop_w = int(w * occ_ratio * 0.6)
                mask[:, w - crop_w:] = 0
                visible_bbox = [x, y, max(1, min(bw, w - crop_w - x)), bh]
        
        else:
            visible_bbox = list(bbox)
        
        # Ensure valid bbox
        visible_bbox = [max(0, float(v)) for v in visible_bbox]
        if visible_bbox[2] <= 0 or visible_bbox[3] <= 0:
            visible_bbox = list(bbox)
        
        return mask, visible_bbox
    
    def _compute_occlusion_rate(self, amodal_bbox: List[float], 
                                visible_bbox: List[float]) -> float:
        """Compute occlusion rate from areas"""
        amodal_area = amodal_bbox[2] * amodal_bbox[3]
        visible_area = visible_bbox[2] * visible_bbox[3]
        
        if amodal_area <= 0:
            return 0.0
        
        return max(0.0, min(1.0, 1.0 - visible_area / amodal_area))
    
    def generate_dataset(self):
        """Generate synthetic amodal dataset"""
        print(f"🎨 Generating {self.target_samples} synthetic instances...\n")
        
        # Shuffle annotations
        random.shuffle(self.person_anns)
        
        synthetic_data = []
        images_used = set()
        
        with tqdm(total=self.target_samples, desc="Generating") as pbar:
            for ann_idx, ann in enumerate(self.person_anns):
                if len(synthetic_data) >= self.target_samples:
                    break
                
                img_id = ann['image_id']
                img_info = self.coco.loadImgs([img_id])[0]
                
                # Load image
                img_path = os.path.join(self.coco_images_dir, img_info['file_name'])
                if not os.path.exists(img_path):
                    continue
                
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Get bbox
                amodal_bbox = ann['bbox']  # [x, y, w, h]
                
                # Choose occlusion type
                occ_type = random.choices(
                    [ot[0] for ot in self.occlusion_types],
                    weights=[ot[1] for ot in self.occlusion_types]
                )[0]
                
                # Generate occlusion
                mask, visible_bbox = self._get_occlusion_mask(
                    image.shape, amodal_bbox, occ_type
                )
                
                # Compute occlusion rate
                occ_rate = self._compute_occlusion_rate(amodal_bbox, visible_bbox)
                
                # No minimum filter - include all samples (even 0% occlusion)
                
                # Filter: visible bbox must be reasonable
                if visible_bbox[2] < 20 or visible_bbox[3] < 30:
                    continue
                
                # Create synthetic instance
                instance = {
                    'image_id': img_id,
                    'file_name': img_info['file_name'],
                    'image_width': img_info['width'],
                    'image_height': img_info['height'],
                    'visible_bbox': visible_bbox,
                    'amodal_bbox': amodal_bbox,
                    'occlusion_rate': float(occ_rate),
                    'occlusion_type': occ_type,
                    'category': 'person'
                }
                
                synthetic_data.append(instance)
                images_used.add(img_id)
                
                pbar.update(1)
                pbar.set_postfix({
                    'images': len(images_used),
                    'occ_type': occ_type,
                    'occ_rate': f'{occ_rate:.2f}'
                })
        
        print(f"\n✅ Generated {len(synthetic_data)} instances from {len(images_used)} images")
        
        # Statistics
        self._print_statistics(synthetic_data)
        
        # Save annotations
        output_file = os.path.join(self.output_dir, 'synthetic_amodal_annotations.json')
        with open(output_file, 'w') as f:
            json.dump(synthetic_data, f, indent=2)
        
        print(f"\n💾 Saved to: {output_file}")
        
        return synthetic_data
    
    def _print_statistics(self, data: List[Dict]):
        """Print dataset statistics"""
        print(f"\n{'='*80}")
        print("📊 DATASET STATISTICS")
        print(f"{'='*80}")
        
        # Occlusion rates
        occ_rates = [d['occlusion_rate'] for d in data]
        print(f"\n🔍 Occlusion Rates:")
        print(f"   Mean: {np.mean(occ_rates):.2f}")
        print(f"   Median: {np.median(occ_rates):.2f}")
        print(f"   Range: [{np.min(occ_rates):.2f}, {np.max(occ_rates):.2f}]")
        
        # Occlusion types
        occ_types = {}
        for d in data:
            occ_types[d['occlusion_type']] = occ_types.get(d['occlusion_type'], 0) + 1
        
        print(f"\n📦 Occlusion Types:")
        for occ_type, count in sorted(occ_types.items(), key=lambda x: -x[1]):
            pct = count / len(data) * 100
            print(f"   {occ_type:15s}: {count:5d} ({pct:5.1f}%)")
        
        # Expansion ratios
        expansions = []
        for d in data:
            amodal_area = d['amodal_bbox'][2] * d['amodal_bbox'][3]
            visible_area = d['visible_bbox'][2] * d['visible_bbox'][3]
            if visible_area > 0:
                expansions.append(amodal_area / visible_area)
        
        print(f"\n📏 Amodal Expansion (amodal_area / visible_area):")
        print(f"   Mean: {np.mean(expansions):.2f}x")
        print(f"   Median: {np.median(expansions):.2f}x")
        print(f"   Range: [{np.min(expansions):.2f}x, {np.max(expansions):.2f}x]")
        
        print(f"{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Synthetic Amodal Dataset')
    parser.add_argument('--coco-images', default='coco/train2017',
                        help='Path to COCO images (train2017)')
    parser.add_argument('--coco-ann', default='coco/annotations/instances_train2017.json',
                        help='Path to COCO annotations')
    parser.add_argument('--output-dir', default='synthetic_amodal_dataset',
                        help='Output directory')
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Number of instances to generate')
    parser.add_argument('--bottom-ratio', type=float, default=0.40,
                        help='Ratio of bottom occlusions (tables/desks)')
    parser.add_argument('--side-ratio', type=float, default=0.30,
                        help='Ratio of side occlusions (walls/doorframes)')
    parser.add_argument('--corner-ratio', type=float, default=0.20,
                        help='Ratio of corner occlusions (furniture)')
    parser.add_argument('--out-of-frame-ratio', type=float, default=0.10,
                        help='Ratio of out-of-frame occlusions')
    parser.add_argument('--no-occlusion-ratio', type=float, default=0.10,
                        help='Ratio of minimal occlusion (10%% for baseline)')
    args = parser.parse_args()
    
    # Generate dataset
    generator = SyntheticAmodalGenerator(
        coco_images_dir=args.coco_images,
        coco_ann_file=args.coco_ann,
        output_dir=args.output_dir,
        target_samples=args.num_samples,
        bottom_ratio=args.bottom_ratio,
        side_ratio=args.side_ratio,
        corner_ratio=args.corner_ratio,
        out_of_frame_ratio=args.out_of_frame_ratio,
        no_occlusion_ratio=args.no_occlusion_ratio
    )
    
    synthetic_data = generator.generate_dataset()
    
    print(f"\n{'='*80}")
    print("✅ DONE!")
    print(f"{'='*80}")
    print(f"📁 Dataset saved to: {args.output_dir}/")
    print(f"📊 Instances: {len(synthetic_data)}")
    print(f"\n💡 Next steps:")
    print(f"   1. Create dataset loader for this annotation format")
    print(f"   2. Train amodal head on this data")
    print(f"   3. Validate on real occluded images")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()