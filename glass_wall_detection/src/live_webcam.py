#!/usr/bin/env python3
"""
Live webcam inference for glass wall detection.
Draws bounding boxes on the video stream in real time.
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np
import torch

# Add DFINE to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    sys.path.insert(0, project_root)

# Import src module first to trigger all registrations
import src  # noqa: F401

from test import load_trained_model, postprocess_predictions, draw_detections


def preprocess_frame(frame_bgr, image_size=640, color_space='bgr'):
    """Preprocess a frame for DFINE with 640x640 resize."""
    orig_h, orig_w = frame_bgr.shape[:2]

    resized = cv2.resize(frame_bgr, (image_size, image_size))
    if color_space == 'rgb':
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = resized.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std

    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor, (orig_w, orig_h)


def parse_args():
    parser = argparse.ArgumentParser(description='Live webcam inference for glass wall detection')
    parser.add_argument('--checkpoint', default='../models/dfine.pth',
                        help='Path to trained checkpoint')
    parser.add_argument('--config', default='../models/dfine_hgnetv2_x_obj2coco.yml',
                        help='DFINE config file')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--camera-id', type=int, default=1,
                        help='Webcam device ID')
    parser.add_argument('--width', type=int, default=640,
                        help='Capture width')
    parser.add_argument('--height', type=int, default=640,
                        help='Capture height')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Confidence threshold for detections')
    parser.add_argument('--glass-only', action='store_true',
                        help='Only draw glass detections (default: show all classes)')
    parser.add_argument('--color-space', choices=['bgr', 'rgb'], default='rgb',
                        help='Input color space for preprocessing (default: bgr)')
    parser.add_argument('--score-mode', choices=['sigmoid', 'softmax'], default='softmax',
                        help='Scoring mode for logits (default: sigmoid)')
    parser.add_argument('--flip', action='store_true',
                        help='Flip frame horizontally')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable GUI display (headless mode)')
    parser.add_argument('--output', default=None,
                        help='Optional output video file path (mp4/avi)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("GLASS WALL DETECTION - LIVE WEBCAM")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Color space: {args.color_space}")
    print(f"Score mode: {args.score_mode}")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, postprocessor = load_trained_model(args.checkpoint, args.config, device)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "Glass Wall Detection (Press Q to quit)"
    display_enabled = not args.no_display
    if display_enabled:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error:
            print("⚠️  OpenCV GUI not available. Running in headless mode.")
            display_enabled = False

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, 20.0, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {args.output}")

    last_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️  Failed to read frame")
            time.sleep(0.1)
            continue

        if args.flip:
            frame = cv2.flip(frame, 1)

        # Resize frame to 640x640 for consistent drawing and input
        frame_640 = cv2.resize(frame, (640, 640))

        input_tensor, orig_size = preprocess_frame(frame_640, color_space=args.color_space)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            outputs = model(input_tensor)

        # Postprocess in 640x640 space
        if args.score_mode == 'softmax':
            pred_logits = outputs['pred_logits'][0]
            pred_boxes = outputs['pred_boxes'][0]
            probs = torch.softmax(pred_logits, dim=-1)
            max_scores, labels = probs.max(dim=-1)
            mask = max_scores > args.confidence
            boxes = pred_boxes[mask]
            scores = max_scores[mask]
            labels = labels[mask]

            # Convert boxes to pixel xyxy
            if boxes.numel() > 0:
                cx, cy, w, h = boxes.unbind(dim=-1)
                x1 = (cx - w / 2) * 640
                y1 = (cy - h / 2) * 640
                x2 = (cx + w / 2) * 640
                y2 = (cy + h / 2) * 640
                boxes = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()
            else:
                boxes = np.array([])
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()

            # All detections (no threshold)
            all_boxes, all_scores, all_labels = postprocess_predictions(
                outputs, postprocessor, (640, 640), 0.0, device
            )
        else:
            boxes, scores, labels = postprocess_predictions(
                outputs, postprocessor, (640, 640), args.confidence, device
            )
            all_boxes, all_scores, all_labels = postprocess_predictions(
                outputs, postprocessor, (640, 640), 0.0, device
            )
        if len(all_scores) > 0:
            topk = min(10, len(all_scores))
            top_idx = np.argsort(-all_scores)[:topk]
            print("\nTop detections (no threshold):")
            for i in top_idx:
                print(f"  class={int(all_labels[i])} score={all_scores[i]:.4f} "
                      f"box=[{all_boxes[i][0]:.0f},{all_boxes[i][1]:.0f},"
                      f"{all_boxes[i][2]:.0f},{all_boxes[i][3]:.0f}]")

        vis_frame = draw_detections(frame_640, boxes, scores, labels, show_all=not args.glass_only)

        # FPS display
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
        last_time = now
        cv2.putText(
            vis_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        if writer:
            writer.write(vis_frame)

        if display_enabled:
            cv2.imshow(window_name, vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
    if display_enabled:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
