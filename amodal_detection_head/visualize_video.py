#!/usr/bin/env python3
"""
Process video with DFINE + Amodal predictions
Shows both visible detections and amodal expansions
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'amodal_detection_head'))
sys.path.insert(0, str(PROJECT_ROOT / 'glass_wall_detection' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'segmentation_sivert'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

from amodal_head import AmodalOffsetHead, extract_roi_features


def load_amodal_model(config_path, checkpoint_path, device):
    """Load DFINE + amodal model"""
    from model_architecture import SegmentationHead, DFineWithSegmentation
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'dfine_model' not in checkpoint or 'amodal_head' not in checkpoint:
        raise ValueError("Need checkpoint with both dfine_model and amodal_head!")
    
    dfine_state = checkpoint['dfine_model']
    amodal_state = checkpoint['amodal_head']
    
    try:
        from src.core import YAMLConfig
    except:
        from core import YAMLConfig
    
    cfg = YAMLConfig(str(config_path))
    base_model = cfg.model
    
    base_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640)
        backbone_features = base_model.backbone(dummy_input)
        backbone_channels = [feat.shape[1] for feat in backbone_features]
    
    feature_dim = 256
    for key in dfine_state.keys():
        if 'seg_head.fpn.lateral_convs.0.weight' in key:
            feature_dim = dfine_state[key].shape[0]
            break
    
    seg_head = SegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=7,
        feature_dim=feature_dim,
        dropout_rate=0.1
    )
    
    dfine_model = DFineWithSegmentation(
        dfine_model=base_model,
        seg_head=seg_head,
        freeze_detection=False
    )
    
    dfine_model.load_state_dict(dfine_state, strict=False)
    
    args = checkpoint.get('args', {})
    amodal_head = AmodalOffsetHead(
        in_channels=backbone_channels[-1],
        hidden_dim=args.get('hidden_dim', 512),
        roi_size=args.get('roi_size', 7)
    )
    
    amodal_head.load_state_dict(amodal_state)
    
    return dfine_model.to(device), amodal_head.to(device), args.get('roi_size', 7)


def detect_with_dfine(model, image_tensor, conf_threshold=0.3, nms_threshold=0.65):
    """Run DFINE detection"""
    with torch.no_grad():
        outputs = model(image_tensor)
    
    pred_logits = outputs['pred_logits'][0]
    pred_boxes = outputs['pred_boxes'][0]
    
    num_classes = pred_logits.shape[-1]
    scores = pred_logits.sigmoid()
    
    # Top-k selection
    scores_flat = scores.flatten()
    topk_values, topk_indices = torch.topk(
        scores_flat,
        k=min(300, scores_flat.numel())
    )
    
    labels = torch.remainder(topk_indices, num_classes)
    query_indices = topk_indices // num_classes
    boxes_cxcywh = pred_boxes[query_indices]
    scores = topk_values
    
    # Filter by confidence
    keep = scores > conf_threshold
    boxes_cxcywh = boxes_cxcywh[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # NMS
    if len(boxes_cxcywh) > 0:
        from torchvision.ops import nms, box_convert
        
        boxes_xyxy = box_convert(boxes_cxcywh, 'cxcywh', 'xyxy')
        
        keep_indices = []
        for class_id in torch.unique(labels):
            class_mask = labels == class_id
            class_boxes = boxes_xyxy[class_mask]
            class_scores = scores[class_mask]
            class_indices = torch.where(class_mask)[0]
            
            nms_keep = nms(class_boxes.float(), class_scores.float(), nms_threshold)
            keep_indices.append(class_indices[nms_keep])
        
        if keep_indices:
            keep_indices = torch.cat(keep_indices)
            boxes_cxcywh = boxes_cxcywh[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
    
    return boxes_cxcywh, scores, labels


def predict_amodal_boxes(amodal_head, visible_boxes_norm, feature_map, roi_size):
    """Predict amodal boxes"""
    if len(visible_boxes_norm) == 0:
        return visible_boxes_norm.clone()
    
    roi_features = extract_roi_features(feature_map, visible_boxes_norm, roi_size)
    
    with torch.no_grad():
        predictions = amodal_head(roi_features, visible_boxes_norm)
    
    return predictions['amodal_boxes']


def normalize_boxes(boxes_cxcywh):
    """Convert cxcywh to normalized xyxy"""
    from torchvision.ops import box_convert
    boxes_xyxy = box_convert(boxes_cxcywh, 'cxcywh', 'xyxy')
    return boxes_xyxy.clamp(0, 1)


def denormalize_boxes(boxes_norm, w, h):
    """Convert normalized xyxy to pixel coords"""
    boxes_px = boxes_norm.clone()
    boxes_px[:, [0, 2]] *= w
    boxes_px[:, [1, 3]] *= h
    return boxes_px


def draw_detections(frame, visible_boxes, amodal_boxes, scores, labels):
    """Draw both visible and amodal boxes on frame"""
    # Filter for person only (label 0)
    person_mask = labels == 0
    if person_mask.sum() == 0:
        return frame
    
    visible_boxes = visible_boxes[person_mask]
    amodal_boxes = amodal_boxes[person_mask]
    scores = scores[person_mask]
    
    for vis_box, amod_box, score in zip(visible_boxes, amodal_boxes, scores):
        # Compute occlusion
        vis_area = (vis_box[2] - vis_box[0]) * (vis_box[3] - vis_box[1])
        amod_area = (amod_box[2] - amod_box[0]) * (amod_box[3] - amod_box[1])
        occlusion = max(0, 1 - vis_area / (amod_area + 1e-6))
        
        # Colors based on occlusion
        if occlusion < 0.15:
            vis_color, amod_color = (0, 255, 0), (0, 200, 0)  # Green
        elif occlusion < 0.35:
            vis_color, amod_color = (0, 255, 255), (0, 200, 200)  # Yellow
        else:
            vis_color, amod_color = (0, 0, 255), (0, 0, 200)  # Red
        
        # Draw amodal box (SOLID - predicted full extent)
        x1, y1, x2, y2 = map(int, amod_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), amod_color, 3)
        
        # Draw visible box (DASHED - detected part)
        x1, y1, x2, y2 = map(int, vis_box)
        
        # Dashed rectangle
        dash_length = 10
        # Top
        for i in range(x1, x2, dash_length * 2):
            cv2.line(frame, (i, y1), (min(i + dash_length, x2), y1), vis_color, 2)
        # Bottom
        for i in range(x1, x2, dash_length * 2):
            cv2.line(frame, (i, y2), (min(i + dash_length, x2), y2), vis_color, 2)
        # Left
        for i in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, i), (x1, min(i + dash_length, y2)), vis_color, 2)
        # Right
        for i in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x2, i), (x2, min(i + dash_length, y2)), vis_color, 2)
        
        # Label
        label_text = f'person {score:.2f} | Occ: {occlusion:.2f}'
        
        # Get amodal box top-left for label
        amod_x1, amod_y1 = int(amod_box[0]), int(amod_box[1])
        
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (amod_x1, amod_y1 - text_h - 10), 
                     (amod_x1 + text_w + 10, amod_y1), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (amod_x1 + 5, amod_y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Legend
    legend_y = 30
    cv2.rectangle(frame, (10, 10), (350, 110), (0, 0, 0), -1)
    cv2.putText(frame, "SOLID = Amodal (predicted)", (20, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "DASHED = Visible (detected)", (20, legend_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "GREEN = Low occ | YELLOW = Med | RED = High", (20, legend_y + 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def process_video(input_path, output_path, dfine_model, amodal_head, roi_size, device, conf_threshold):
    """Process video frame by frame"""
    import subprocess
    import tempfile
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 Video info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f}s\n")
    
    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_pattern = os.path.join(temp_dir, 'frame_%06d.png')
    
    # Preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    pbar = tqdm(total=total_frames, desc="Processing")
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = (img_norm - mean) / std
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Detect
        boxes_cxcywh, scores, labels = detect_with_dfine(dfine_model, img_tensor, conf_threshold)
        
        if len(boxes_cxcywh) > 0:
            # Get person detections
            person_mask = labels == 0
            if person_mask.sum() > 0:
                # Normalize boxes
                visible_boxes_norm = normalize_boxes(boxes_cxcywh)
                
                # Extract features
                if hasattr(dfine_model, 'dfine_model'):
                    features = dfine_model.dfine_model.backbone(img_tensor)
                else:
                    features = dfine_model.backbone(img_tensor)
                
                if isinstance(features, (list, tuple)):
                    feature_map = features[-1]
                else:
                    feature_map = features
                
                # Predict amodal
                amodal_boxes_norm = predict_amodal_boxes(
                    amodal_head, visible_boxes_norm, feature_map, roi_size
                )
                
                # Denormalize to frame size
                visible_boxes_px = denormalize_boxes(visible_boxes_norm, width, height)
                amodal_boxes_px = denormalize_boxes(amodal_boxes_norm, width, height)
                
                # Draw
                frame = draw_detections(
                    frame,
                    visible_boxes_px.detach().cpu().numpy(),
                    amodal_boxes_px.detach().cpu().numpy(),
                    scores.detach().cpu().numpy(),
                    labels.detach().cpu().numpy()
                )
        
        # Save frame
        frame_path = os.path.join(temp_dir, f'frame_{frame_idx:06d}.png')
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Use ffmpeg to create video
    print("\n🎬 Encoding video with ffmpeg...")
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-framerate', str(fps),
        '-i', frame_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',  # High quality
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Output saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e.stderr.decode()}")
        raise
    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir)
        print(f"🗑️  Cleaned up temp files")


def main():
    parser = argparse.ArgumentParser(description='Process video with amodal detection')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--config', default='models/dfine_hgnetv2_x_obj2coco.yml',
                        help='DFINE config')
    parser.add_argument('--checkpoint', default='outputs/amodal_synthetic/best_model.pth',
                        help='Amodal checkpoint')
    parser.add_argument('--conf-threshold', type=float, default=0.6,
                        help='Detection confidence threshold')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print("🎬 VIDEO AMODAL DETECTION")
    print(f"{'='*80}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    print(f"{'='*80}")
    
    # Load models
    print("\n📦 Loading models...")
    dfine_model, amodal_head, roi_size = load_amodal_model(
        args.config, args.checkpoint, device
    )
    dfine_model.eval()
    amodal_head.eval()
    print("✅ Models loaded")
    
    # Process video
    process_video(
        args.input,
        args.output,
        dfine_model,
        amodal_head,
        roi_size,
        device,
        args.conf_threshold
    )
    
    print(f"\n{'='*80}")
    print("🎉 Done!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()