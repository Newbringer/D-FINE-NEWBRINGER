#!/usr/bin/env python3
"""
Fixed TensorRT inference script for DFINE segmentation model.
Properly handles DFINE's coordinate system.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# COCO classes
COCO_80_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

COCO_81_CLASSES = COCO_80_CLASSES + ['glass_wall']

SEGMENTATION_CLASSES = [
    "background", "head", "torso", "upper_arms", 
    "lower_arms", "upper_legs", "lower_legs"
]

SEGMENTATION_COLORS = {
    0: (0, 0, 0),        # background
    1: (0, 0, 255),      # head - red
    2: (0, 165, 255),    # torso - orange
    3: (0, 255, 255),    # upper_arms - yellow
    4: (255, 0, 255),    # lower_arms - magenta
    5: (0, 255, 0),      # upper_legs - green
    6: (255, 0, 0),      # lower_legs - blue
}


class HostDeviceMem:
    """Host and device memory pair for TensorRT"""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem


class TensorRTEngine:
    """TensorRT engine wrapper for DFINE segmentation model"""
    
    def __init__(self, engine_path: str, input_size: Tuple[int, int] = (640, 640)):
        self.engine_path = engine_path
        self.input_size = input_size
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        print(f"Loading TensorRT engine: {engine_path}")
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        
        print("Allocating buffers...")
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        # Auto-detect number of classes
        self.num_classes = 80  # Will update if we see class >= 80
        self.class_names = COCO_80_CLASSES
        
        print("✓ TensorRT engine loaded successfully")
        self._print_engine_info()
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        return engine
    
    def _allocate_buffers(self):
        """Allocate host and device buffers"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        use_new_api = hasattr(self.engine, 'num_io_tensors')
        
        if use_new_api:
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                shape = self.engine.get_tensor_shape(tensor_name)
                
                size = abs(np.prod(shape))
                if size <= 0:
                    size = 1
                
                host_mem = cuda.pagelocked_empty(int(size), dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(device_mem))
                
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    inputs.append(HostDeviceMem(host_mem, device_mem))
                else:
                    outputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            for binding in range(self.engine.num_bindings):
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                shape = self.engine.get_binding_shape(binding)
                
                size = abs(np.prod(shape))
                if size <= 0:
                    size = 1
                
                host_mem = cuda.pagelocked_empty(int(size), dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(device_mem))
                
                if self.engine.binding_is_input(binding):
                    inputs.append(HostDeviceMem(host_mem, device_mem))
                else:
                    outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
    
    def _print_engine_info(self):
        """Print engine information"""
        print("\n" + "="*80)
        print("ENGINE INFORMATION")
        print("="*80)
        print(f"Number of classes: {self.num_classes}")
        print(f"Model input size: {self.input_size}")
        
        use_new_api = hasattr(self.engine, 'num_io_tensors')
        
        if use_new_api:
            print(f"Total tensors: {self.engine.num_io_tensors}")
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(tensor_name)
                dtype = self.engine.get_tensor_dtype(tensor_name)
                mode = self.engine.get_tensor_mode(tensor_name)
                
                if mode == trt.TensorIOMode.INPUT:
                    print(f"  Input: {tensor_name}, shape: {shape}, dtype: {dtype}")
                else:
                    print(f"  Output: {tensor_name}, shape: {shape}, dtype: {dtype}")
        else:
            print(f"Total bindings: {self.engine.num_bindings}")
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                dtype = self.engine.get_binding_dtype(i)
                is_input = self.engine.binding_is_input(i)
                
                binding_type = "Input" if is_input else "Output"
                print(f"  {binding_type}: {name}, shape: {shape}, dtype: {dtype}")
        
        print("="*80 + "\n")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for DFINE inference.
        
        IMPORTANT: orig_size tells the model what coordinate space to output boxes in!
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        h, w = self.input_size
        resized = cv2.resize(image_rgb, (w, h))
        
        # Normalize (ImageNet normalization)
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Convert to CHW format
        input_tensor = normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, 0).astype(np.float32)
        
        # CRITICAL: Original size tells model to output boxes in THIS coordinate space
        # Format: [[width, height]] in int64
        orig_size = np.array([[image.shape[1], image.shape[0]]], dtype=np.int64)
        
        return input_tensor, orig_size
    
    def infer(self, image_tensor: np.ndarray, orig_size: np.ndarray) -> dict:
        """Run inference"""
        # Copy inputs to device
        np.copyto(self.inputs[0].host, image_tensor.ravel())
        np.copyto(self.inputs[1].host, orig_size.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        # Run inference
        use_new_api = hasattr(self.engine, 'num_io_tensors')
        
        if use_new_api:
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(tensor_name)
                
                if mode == trt.TensorIOMode.INPUT:
                    idx = 0 if 'images' in tensor_name else 1
                    self.context.set_tensor_address(tensor_name, int(self.inputs[idx].device))
                else:
                    output_names = ['labels', 'boxes', 'scores', 'seg_probs', 'seg_preds']
                    try:
                        idx = output_names.index(tensor_name)
                    except ValueError:
                        idx = len(self.outputs) - 1
                    
                    if idx < len(self.outputs):
                        self.context.set_tensor_address(tensor_name, int(self.outputs[idx].device))
            
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy outputs back to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        
        self.stream.synchronize()
        
        # Parse outputs
        labels = self.outputs[0].host.copy().reshape(1, 300)
        boxes = self.outputs[1].host.copy().reshape(1, 300, 4)
        scores = self.outputs[2].host.copy().reshape(1, 300)
        
        seg_probs = None
        seg_preds = None
        
        if len(self.outputs) > 3:
            try:
                seg_probs = self.outputs[3].host.copy().reshape(1, 7, 640, 640)
            except:
                pass
        
        if len(self.outputs) > 4:
            try:
                seg_preds = self.outputs[4].host.copy().reshape(1, 640, 640)
            except:
                pass
        
        return {
            'labels': labels,
            'boxes': boxes,
            'scores': scores,
            'seg_probs': seg_probs,
            'seg_preds': seg_preds,
        }
    
    def postprocess(self, results: dict, confidence_threshold: float = 0.3,
                    orig_width: int = 1280, orig_height: int = 720) -> dict:
        """
        Postprocess DFINE outputs.
        
        CRITICAL: DFINE outputs boxes in the coordinate space specified by orig_size input.
        Since we passed the original image dimensions, boxes are ALREADY in pixel coordinates
        for the original image - NO SCALING NEEDED!
        
        Args:
            results: Raw inference results
            confidence_threshold: Minimum confidence
            orig_width: Original image width (for reference)
            orig_height: Original image height (for reference)
        """
        # Extract arrays
        labels = results['labels'][0]  # (300,)
        boxes = results['boxes'][0]    # (300, 4) - ALREADY in original image pixel coords!
        scores = results['scores'][0]  # (300,)
        
        print(f"\n{'='*80}")
        print("POSTPROCESSING")
        print(f"{'='*80}")
        print(f"Original image size: {orig_width}x{orig_height}")
        print(f"Raw candidates: {len(scores)}")
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Print some sample boxes to understand coordinate space
        if len(boxes) > 0:
            print(f"\nSample raw boxes (first 3):")
            for i in range(min(3, len(boxes))):
                print(f"  Box {i}: [{boxes[i][0]:.2f}, {boxes[i][1]:.2f}, {boxes[i][2]:.2f}, {boxes[i][3]:.2f}] score={scores[i]:.4f}")
        
        # Update num_classes if needed
        max_label = int(labels.max())
        if max_label >= 80 and self.num_classes == 80:
            self.num_classes = 81
            self.class_names = COCO_81_CLASSES
            print(f"⚠️  Detected class {max_label}, switching to 81-class mode")
        
        # STEP 1: Remove NaN/Inf
        finite_mask = (
            np.isfinite(scores) & 
            np.isfinite(boxes).all(axis=1) & 
            np.isfinite(labels)
        )
        print(f"After NaN/Inf removal: {finite_mask.sum()}")
        
        # STEP 2: Confidence threshold
        finite_mask &= (scores >= confidence_threshold)
        print(f"After confidence >= {confidence_threshold}: {finite_mask.sum()}")
        
        # Apply mask
        boxes = boxes[finite_mask].copy()
        scores = scores[finite_mask].copy()
        labels = labels[finite_mask].copy()
        
        if len(boxes) == 0:
            print(f"{'='*80}\n")
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([]),
                'seg_preds': results['seg_preds'][0] if results['seg_preds'] is not None else None
            }
        
        # STEP 3: Validate boxes are in reasonable range for the image
        # Boxes should be in [0, width] x [0, height] range
        print(f"\nBox coordinate analysis:")
        print(f"  X range: [{boxes[:, [0,2]].min():.2f}, {boxes[:, [0,2]].max():.2f}] (should be 0-{orig_width})")
        print(f"  Y range: [{boxes[:, [1,3]].min():.2f}, {boxes[:, [1,3]].max():.2f}] (should be 0-{orig_height})")
        
        # Check if boxes are in pixel coordinates (expected) or normalized
        max_coord = max(boxes.max(), abs(boxes.min()))
        if max_coord <= 2.0:
            print(f"  ⚠️  Boxes appear NORMALIZED (max={max_coord:.2f})")
            print(f"  Converting to pixel coordinates...")
            # Convert from normalized to pixel
            boxes[:, [0, 2]] *= orig_width
            boxes[:, [1, 3]] *= orig_height
        else:
            print(f"  ✓ Boxes appear to be in PIXEL coordinates (max={max_coord:.2f})")
        
        # STEP 4: Clamp boxes to image boundaries and validate
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_height)
        
        # Ensure x2 > x1 and y2 > y1
        valid_boxes = (
            (boxes[:, 2] > boxes[:, 0] + 1) &  # Width > 1
            (boxes[:, 3] > boxes[:, 1] + 1)    # Height > 1
        )
        
        print(f"After box validation: {valid_boxes.sum()}")
        
        boxes = boxes[valid_boxes].copy()
        scores = scores[valid_boxes].copy()
        labels = labels[valid_boxes].copy()
        
        # STEP 5: Validate class IDs
        valid_classes = (labels >= 0) & (labels < self.num_classes)
        print(f"After class ID validation: {valid_classes.sum()}")
        
        boxes = boxes[valid_classes].copy()
        scores = scores[valid_classes].copy()
        labels = labels[valid_classes].copy()
        
        # Summary
        print(f"\n✅ FINAL: {len(boxes)} valid detections")
        if len(boxes) > 0:
            print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Box range: [{boxes.min():.2f}, {boxes.max():.2f}]")
            print(f"  Classes: {np.unique(labels.astype(int)).tolist()}")
            print(f"\nFinal boxes (first 3):")
            for i in range(min(3, len(boxes))):
                w = boxes[i][2] - boxes[i][0]
                h = boxes[i][3] - boxes[i][1]
                print(f"  Box {i}: x=[{boxes[i][0]:.1f},{boxes[i][2]:.1f}] y=[{boxes[i][1]:.1f},{boxes[i][3]:.1f}] size={w:.1f}x{h:.1f}")
        print(f"{'='*80}\n")
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'seg_preds': results['seg_preds'][0] if results['seg_preds'] is not None else None
        }


def draw_detections(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, 
                    labels: np.ndarray, class_names: List[str], 
                    show_labels: bool = True) -> np.ndarray:
    """
    Draw bounding boxes on frame.
    Boxes are assumed to be in pixel coordinates: [x1, y1, x2, y2]
    """
    vis_frame = frame.copy()
    
    if len(boxes) == 0:
        return vis_frame
    
    print(f"\n🎨 Drawing {len(boxes)} detections")
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Boxes are already in pixel coordinates, just convert to int
        x1, y1, x2, y2 = box.astype(int)
        
        # Validate
        if x2 <= x1 or y2 <= y1:
            print(f"  ⚠️  Skipping invalid box {i}: ({x1},{y1})-({x2},{y2})")
            continue
        
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            print(f"  ⚠️  Box {i} outside frame bounds: ({x1},{y1})-({x2},{y2})")
        
        class_idx = int(label)
        
        # Choose color
        if class_idx == 80:  # glass_wall
            color = (0, 0, 255)  # RED
        elif class_idx == 0:  # person
            color = (0, 255, 0)  # GREEN
        else:
            color = (255, 255, 0)  # CYAN for others
        
        # Draw box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if show_labels:
            class_name = class_names[class_idx] if class_idx < len(class_names) else f'class_{class_idx}'
            label_text = f'{class_name}: {score:.2f}'
            
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background for text
            cv2.rectangle(vis_frame, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(vis_frame, label_text, (x1, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        box_w = x2 - x1
        box_h = y2 - y1
        print(f"  ✓ {class_name} @ ({x1},{y1})-({x2},{y2}) size={box_w}x{box_h} conf={score:.3f}")
    
    return vis_frame


def overlay_segmentation(frame: np.ndarray, seg_preds: np.ndarray, 
                        person_boxes: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay segmentation on detected person regions"""
    if seg_preds is None:
        print("  ⚠️  No segmentation predictions available")
        return frame
    
    if len(person_boxes) == 0:
        print("  ⚠️  No person boxes found for segmentation")
        return frame
    
    print(f"\n🎨 Overlaying segmentation on {len(person_boxes)} person(s)")
    print(f"  Segmentation mask shape: {seg_preds.shape}")
    print(f"  Segmentation mask range: [{seg_preds.min()}, {seg_preds.max()}]")
    print(f"  Unique classes in mask: {np.unique(seg_preds)}")
    
    # Resize segmentation to frame size
    seg_mask = cv2.resize(seg_preds.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    # Create full-frame overlay (not just person regions)
    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    # Apply colors to all segmentation classes
    for class_id, color in SEGMENTATION_COLORS.items():
        if class_id == 0:  # Skip background
            continue
        mask = seg_mask == class_id
        overlay[mask] = color
        pixel_count = mask.sum()
        if pixel_count > 0:
            print(f"  Class {class_id} ({SEGMENTATION_CLASSES[class_id]}): {pixel_count} pixels")
    
    # Check if we actually have any segmentation
    non_bg_mask = seg_mask > 0
    if not non_bg_mask.any():
        print("  ⚠️  Segmentation mask is all background!")
        return frame
    
    print(f"  Total segmentation pixels: {non_bg_mask.sum()}")
    
    # Blend with original frame
    blended = frame.copy()
    blended[non_bg_mask] = (frame[non_bg_mask] * (1 - alpha) + overlay[non_bg_mask] * alpha).astype(np.uint8)
    
    return blended


def process_webcam(engine: TensorRTEngine, args):
    """Process webcam input"""
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera_id}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n🎥 Starting webcam inference")
    print(f"Camera ID: {args.camera_id}")
    print(f"Resolution: {actual_width}x{actual_height}")
    print(f"Confidence: {args.confidence}")
    print(f"Model classes: {engine.num_classes}")
    print(f"Segmentation: {'ENABLED' if args.segmentation else 'DISABLED (use --segmentation to enable)'}")
    print("Press 'q' to quit\n")
    
    window_name = "DFINE TensorRT - Press Q to quit"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, 20.0, (actual_width, actual_height))
    
    fps_history = []
    frame_count = 0
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            if args.flip:
                frame = cv2.flip(frame, 1)
            
            # Get actual frame dimensions
            frame_h, frame_w = frame.shape[:2]
            
            # Preprocess
            input_tensor, orig_size = engine.preprocess(frame)
            
            # Inference
            raw_results = engine.infer(input_tensor, orig_size)
            
            # Check segmentation availability on first frame
            if frame_count == 0:
                if raw_results['seg_preds'] is not None:
                    print(f"✓ Segmentation output detected: shape={raw_results['seg_preds'].shape}")
                else:
                    print(f"⚠️  No segmentation output from model")
                    if args.segmentation:
                        print(f"   Model may not have been exported with segmentation enabled")
            
            # Postprocess
            results = engine.postprocess(raw_results, args.confidence, 
                                        orig_width=frame_w, orig_height=frame_h)
            
            # Boxes are ALREADY in pixel coordinates
            boxes = results['boxes']
            
            # Draw detections
            vis_frame = draw_detections(frame, boxes, results['scores'], 
                                       results['labels'], engine.class_names)
            
            # Segmentation overlay
            if args.segmentation:
                if results['seg_preds'] is not None:
                    # Find person boxes
                    if len(boxes) > 0:
                        person_mask = results['labels'] == 0
                        person_boxes = boxes[person_mask]
                        
                        if len(person_boxes) > 0:
                            vis_frame = overlay_segmentation(vis_frame, results['seg_preds'], 
                                                           person_boxes, args.seg_alpha)
                        else:
                            # Still show segmentation even without person detections
                            # (model might segment people below confidence threshold)
                            vis_frame = overlay_segmentation(vis_frame, results['seg_preds'], 
                                                           np.array([]), args.seg_alpha)
                    else:
                        # No detections but might still have segmentation
                        vis_frame = overlay_segmentation(vis_frame, results['seg_preds'], 
                                                       np.array([]), args.seg_alpha)
                else:
                    if frame_count == 0:
                        print("⚠️  Segmentation requested but not available from model")
            
            # FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # Draw stats
            cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Detections: {len(boxes)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Resolution: {frame_w}x{frame_h}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show if segmentation is active
            if args.segmentation and results['seg_preds'] is not None:
                cv2.putText(vis_frame, "Segmentation: ON", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if writer is not None:
                writer.write(vis_frame)
            
            if not args.no_display:
                cv2.imshow(window_name, vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        print(f"\n✓ Processed {frame_count} frames")
        if fps_history:
            print(f"Average FPS: {sum(fps_history)/len(fps_history):.1f}")



def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='TensorRT inference for DFINE (80 or 81 classes)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model
    parser.add_argument('--engine', required=True, help='Path to TensorRT engine file')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                       help='Model input size (height width)')
    
    # Input source
    parser.add_argument('--mode', choices=['webcam', 'video', 'images'], default='webcam')
    parser.add_argument('--input', help='Input video/image directory')
    parser.add_argument('--camera-id', type=int, default=0, help='Webcam device ID')
    
    # Output
    parser.add_argument('--output', help='Output file/directory')
    parser.add_argument('--no-display', action='store_true', help='Disable GUI')
    
    # Webcam
    parser.add_argument('--width', type=int, default=1280, help='Webcam width')
    parser.add_argument('--height', type=int, default=720, help='Webcam height')
    parser.add_argument('--flip', action='store_true', help='Flip horizontally')
    
    # Detection
    parser.add_argument('--confidence', type=float, default=0.3, 
                       help='Confidence threshold')
    
    # Segmentation
    parser.add_argument('--segmentation', action='store_true', 
                       help='Enable person segmentation')
    parser.add_argument('--seg-alpha', type=float, default=0.4, 
                       help='Segmentation opacity')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    print("\n" + "="*80)
    print("DFINE TENSORRT INFERENCE - FIXED COORDINATE HANDLING")
    print("="*80)
    print(f"Engine: {args.engine}")
    print(f"Mode: {args.mode}")
    print(f"Confidence: {args.confidence}")
    print("="*80 + "\n")
    
    # Load engine
    engine = TensorRTEngine(args.engine, tuple(args.input_size))
    
    # Run inference
    try:
        if args.mode == 'webcam':
            process_webcam(engine, args)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✓ Done")


if __name__ == '__main__':
    main()
