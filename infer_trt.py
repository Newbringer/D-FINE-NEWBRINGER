#!/usr/bin/env python3
"""
DFINE Segmentation TensorRT Inference Script

Run inference on video using TensorRT engine for DFINE model with segmentation.
Usage:
    python tensorrt/infer_segmentation_trt.py -e model_segmentation.engine -i input_video.mp4 -o output_video.mp4
"""

import os
import sys
import argparse
import time
import subprocess
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DFINE Segmentation TensorRT Inference')
    parser.add_argument('-e', '--engine', type=str, required=True,
                        help='Path to TensorRT engine file')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('-o', '--output', type=str, default='dfine_segmentation_result_trt.mp4',
                        help='Path to output video (default: dfine_segmentation_result_trt.mp4)')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--seg-alpha', type=float, default=0.6,
                        help='Segmentation overlay alpha (default: 0.6)')
    parser.add_argument('--show-detection', action='store_true', default=True,
                        help='Show detection boxes (default: True)')
    parser.add_argument('--show-segmentation', action='store_true', default=True,
                        help='Show segmentation masks (default: True)')
    parser.add_argument('--resize-factor', type=float, default=1.0,
                        help='Resize input frames by this factor (1.0 = original size)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output FPS (default: same as input)')
    parser.add_argument('--crf', type=int, default=23,
                        help='CRF value for video encoding (lower = better quality, 0-51, default: 23)')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                        help='x264 encoding preset (affects speed and compression, default: medium)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run in benchmark mode (no video output, just measure FPS)')
    parser.add_argument('--benchmark-frames', type=int, default=300,
                        help='Number of frames to process in benchmark mode (default: 300)')
    return parser.parse_args()

def get_body_part_colors():
    """Get colors for each body part class - matching PyTorch script"""
    return {
        0: (0, 0, 0),        # background - black (transparent)
        1: (255, 0, 0),      # head - red
        2: (0, 255, 0),      # torso - green  
        3: (0, 0, 255),      # arms - blue
        4: (255, 255, 0),    # hands - yellow
        5: (255, 0, 255),    # legs - magenta
        6: (0, 255, 255),    # feet - cyan
    }

def get_body_part_names():
    """Get names for each body part class"""
    return {
        0: 'background',
        1: 'head',
        2: 'torso', 
        3: 'arms',
        4: 'hands',
        5: 'legs',
        6: 'feet'
    }

class TensorRTSegmentationEngine:
    """TensorRT engine wrapper for DFINE segmentation inference"""
    def __init__(self, engine_path):
        """Initialize TensorRT engine"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load TensorRT engine
        print(f"Loading TensorRT segmentation engine from: {engine_path}")
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
            
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if not self.engine:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
            
        self.context = self.engine.create_execution_context()
        
        # Print engine information
        print("Segmentation engine loaded successfully!")
        
        # Determine if we're using the new API or old API
        self.use_new_api = hasattr(self.engine, 'num_io_tensors')
        
        if self.use_new_api:
            print(f"Using TensorRT 10.x API with {self.engine.num_io_tensors} IO tensors")
            # Create input and output binding dictionaries
            self.input_bindings = {}
            self.output_bindings = {}
            
            # Process IO tensors
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                shape = self.engine.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)
                
                if mode == trt.TensorIOMode.INPUT:
                    print(f"  Input: {name}, shape: {shape}, dtype: {dtype}")
                    self.input_bindings[name] = {
                        'index': i,
                        'name': name,
                        'shape': shape,
                        'dtype': self.get_numpy_dtype(dtype),
                    }
                else:
                    print(f"  Output: {name}, shape: {shape}, dtype: {dtype}")
                    self.output_bindings[name] = {
                        'index': i,
                        'name': name,
                        'shape': shape,
                        'dtype': self.get_numpy_dtype(dtype),
                    }
        else:
            print(f"Using legacy TensorRT API with {self.engine.num_bindings} bindings")
            # Create input and output binding dictionaries
            self.input_bindings = {}
            self.output_bindings = {}
            
            # Process bindings
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                dtype = self.engine.get_binding_dtype(i)
                
                if self.engine.binding_is_input(i):
                    print(f"  Input {i}: {name}, shape: {shape}, dtype: {dtype}")
                    self.input_bindings[name] = {
                        'index': i,
                        'name': name,
                        'shape': shape,
                        'dtype': self.get_numpy_dtype(dtype),
                    }
                else:
                    print(f"  Output {i}: {name}, shape: {shape}, dtype: {dtype}")
                    self.output_bindings[name] = {
                        'index': i,
                        'name': name,
                        'shape': shape,
                        'dtype': self.get_numpy_dtype(dtype),
                    }
        
        # Allocate device memory and create CUDA stream
        self.cuda_stream = cuda.Stream()
        self.allocate_buffers()
    
    def get_numpy_dtype(self, trt_dtype):
        """Convert TensorRT dtype to numpy dtype"""
        if trt_dtype == trt.float32:
            return np.float32
        elif trt_dtype == trt.float16:
            return np.float16
        elif trt_dtype == trt.int8:
            return np.int8
        elif trt_dtype == trt.int32:
            return np.int32
        elif trt_dtype == trt.int64:
            return np.int64
        elif trt_dtype == trt.bool:
            return np.bool_
        else:
            return np.float32  # Default to float32
    
    def allocate_buffers(self):
        """Allocate device and host memory for inputs and outputs"""
        self.host_inputs = {}
        self.host_outputs = {}
        self.device_inputs = {}
        self.device_outputs = {}
        
        # Allocate memory for inputs
        for name, binding in self.input_bindings.items():
            shape = binding['shape']
            dtype = binding['dtype']
            
            # For dynamic shapes, use a fixed size for allocation
            if -1 in shape:
                # Replace -1 with a reasonable size (e.g., batch size 1)
                shape = tuple(1 if dim == -1 else dim for dim in shape)
                print(f"  Dynamic input shape for {name}, allocating with shape {shape}")
            
            # Calculate total size of the tensor and convert to Python int
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.host_inputs[name] = host_mem
            self.device_inputs[name] = device_mem
        
        # Allocate memory for outputs
        for name, binding in self.output_bindings.items():
            shape = binding['shape']
            dtype = binding['dtype']
            
            # For dynamic shapes, use a reasonable size for allocation
            if -1 in shape:
                # Handle different output types
                if 'seg' in name.lower():
                    # Segmentation outputs - typically [batch, height, width] or [batch, classes, height, width]
                    if len(shape) == 3:  # [batch, H, W]
                        shape = (1, 640, 640)
                    elif len(shape) == 4:  # [batch, classes, H, W]
                        shape = (1, 7, 640, 640)  # 7 classes for pascal person parts
                else:
                    # Detection outputs
                    if len(shape) == 2:  # Likely [batch_size, num_detections]
                        shape = (1, 300)  # Use 300 detections
                    elif len(shape) == 3:  # Likely [batch_size, num_detections, 4] for boxes
                        shape = (1, 300, shape[2])
                print(f"  Dynamic output shape for {name}, allocating with shape {shape}")
            
            # Calculate total size of the tensor and convert to Python int
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.host_outputs[name] = host_mem
            self.device_outputs[name] = device_mem
    
    def infer(self, images, orig_target_sizes):
        """
        Run inference on the TensorRT segmentation engine
        
        Args:
            images: preprocessed image tensor (1,3,H,W) as numpy array
            orig_target_sizes: original image size tensor (1,2) with [width, height] as numpy array
            
        Returns:
            labels, boxes, scores, seg_probs, seg_preds as numpy arrays
        """
        # We're always using batch size 1 for video processing
        batch_size = 1
        
        # Find input tensor names
        image_input_name = None
        size_input_name = None
        
        for name in self.input_bindings.keys():
            if 'image' in name.lower() or 'input' in name.lower():
                image_input_name = name
            elif 'size' in name.lower() or 'target' in name.lower():
                size_input_name = name
        
        # If not found by name, use position
        if image_input_name is None or size_input_name is None:
            input_names = list(self.input_bindings.keys())
            if len(input_names) >= 2:
                image_input_name = input_names[0]
                size_input_name = input_names[1]
        
        # Copy input data to host buffers
        if image_input_name:
            np.copyto(self.host_inputs[image_input_name].reshape(images.size), images.ravel())
        
        if size_input_name:
            if self.host_inputs[size_input_name].dtype != orig_target_sizes.dtype:
                orig_target_sizes_converted = orig_target_sizes.astype(self.host_inputs[size_input_name].dtype)
                np.copyto(self.host_inputs[size_input_name].reshape(orig_target_sizes.size), 
                         orig_target_sizes_converted.ravel())
            else:
                np.copyto(self.host_inputs[size_input_name].reshape(orig_target_sizes.size), 
                         orig_target_sizes.ravel())
        
        # Find output tensor names
        labels_output_name = None
        boxes_output_name = None
        scores_output_name = None
        seg_probs_output_name = None
        seg_preds_output_name = None
        
        for name in self.output_bindings.keys():
            if 'label' in name.lower() or 'class' in name.lower():
                labels_output_name = name
            elif 'box' in name.lower() or 'rect' in name.lower():
                boxes_output_name = name
            elif 'score' in name.lower() or 'conf' in name.lower():
                scores_output_name = name
            elif 'seg_prob' in name.lower() or 'segmentation_prob' in name.lower():
                seg_probs_output_name = name
            elif 'seg_pred' in name.lower() or 'segmentation_pred' in name.lower():
                seg_preds_output_name = name
        
        # If not found by name, use position (labels, boxes, scores, seg_probs, seg_preds)
        if None in [labels_output_name, boxes_output_name, scores_output_name, seg_probs_output_name, seg_preds_output_name]:
            output_names = list(self.output_bindings.keys())
            if len(output_names) >= 5:
                labels_output_name = output_names[0]
                boxes_output_name = output_names[1]
                scores_output_name = output_names[2]
                seg_probs_output_name = output_names[3]
                seg_preds_output_name = output_names[4]
        
        # Execute inference using the appropriate TensorRT API
        if self.use_new_api:
            # TensorRT 10.x API
            
            # Copy input data to device
            for name, host_mem in self.host_inputs.items():
                cuda.memcpy_htod_async(self.device_inputs[name], host_mem, self.cuda_stream)
            
            # Set input shapes and buffers
            for name, binding in self.input_bindings.items():
                if image_input_name == name:
                    self.context.set_input_shape(name, images.shape)
                elif size_input_name == name:
                    self.context.set_input_shape(name, orig_target_sizes.shape)
                self.context.set_tensor_address(name, int(self.device_inputs[name]))
            
            # Set output buffers
            for name in self.output_bindings.keys():
                self.context.set_tensor_address(name, int(self.device_outputs[name]))
            
            # Run inference
            self.context.execute_async_v3(self.cuda_stream.handle)
            
            # Copy outputs from device to host
            for name, device_mem in self.device_outputs.items():
                cuda.memcpy_dtoh_async(self.host_outputs[name], device_mem, self.cuda_stream)
            
        else:
            # Legacy TensorRT API
            
            # Prepare input bindings
            bindings = []
            all_bindings = list(self.input_bindings.values()) + list(self.output_bindings.values())
            all_bindings.sort(key=lambda x: x['index'])
            
            for binding in all_bindings:
                name = binding['name']
                if name in self.input_bindings:
                    bindings.append(int(self.device_inputs[name]))
                else:
                    bindings.append(int(self.device_outputs[name]))
            
            # Copy input data to device
            for name, host_mem in self.host_inputs.items():
                cuda.memcpy_htod_async(self.device_inputs[name], host_mem, self.cuda_stream)
            
            # Run inference
            self.context.execute_async_v2(bindings, self.cuda_stream.handle, None)
            
            # Copy outputs from device to host
            for name, device_mem in self.device_outputs.items():
                cuda.memcpy_dtoh_async(self.host_outputs[name], device_mem, self.cuda_stream)
        
        # Synchronize the stream
        self.cuda_stream.synchronize()
        
        # Get output shapes from the context
        if self.use_new_api:
            try:
                labels_shape = self.context.get_tensor_shape(labels_output_name)
                boxes_shape = self.context.get_tensor_shape(boxes_output_name)
                scores_shape = self.context.get_tensor_shape(scores_output_name)
                seg_probs_shape = self.context.get_tensor_shape(seg_probs_output_name)
                seg_preds_shape = self.context.get_tensor_shape(seg_preds_output_name)
            except:
                # Fallback shapes
                labels_shape = (1, 300)
                boxes_shape = (1, 300, 4)
                scores_shape = (1, 300)
                seg_preds_shape = (1, 640, 640)
                seg_probs_shape = (1, 7, 640, 640)
        else:
            # For legacy API, estimate shapes
            labels_shape = (1, 300)
            boxes_shape = (1, 300, 4)
            scores_shape = (1, 300)
            seg_preds_shape = (1, 640, 640)
            seg_probs_shape = (1, 7, 640, 640)
        
        # Reshape outputs
        try:
            labels = self.host_outputs[labels_output_name].reshape(labels_shape)
            boxes = self.host_outputs[boxes_output_name].reshape(boxes_shape)
            scores = self.host_outputs[scores_output_name].reshape(scores_shape)
            seg_probs = self.host_outputs[seg_probs_output_name].reshape(seg_probs_shape)
            seg_preds = self.host_outputs[seg_preds_output_name].reshape(seg_preds_shape)
        except ValueError as e:
            print(f"Warning: Failed to reshape outputs: {e}")
            # Use default shapes if reshaping fails
            labels = self.host_outputs[labels_output_name].reshape((1, 300))
            boxes = self.host_outputs[boxes_output_name].reshape((1, 300, 4))
            scores = self.host_outputs[scores_output_name].reshape((1, 300))
            seg_preds = self.host_outputs[seg_preds_output_name].reshape((1, 640, 640))
            seg_probs = self.host_outputs[seg_probs_output_name].reshape((1, 7, 640, 640))
        
        # Convert to list of tensors to match PyTorch model output format
        labels_tensor = [torch.tensor(labels[i]) for i in range(1)]
        boxes_tensor = [torch.tensor(boxes[i]) for i in range(1)]
        scores_tensor = [torch.tensor(scores[i]) for i in range(1)]
        seg_probs_tensor = [torch.tensor(seg_probs[i]) for i in range(1)]
        seg_preds_tensor = [torch.tensor(seg_preds[i]) for i in range(1)]
        
        return labels_tensor, boxes_tensor, scores_tensor, seg_probs_tensor, seg_preds_tensor

def process_frame_segmentation_trt(trt_engine, frame_bgr, transforms):
    """
    Process a single frame with the TensorRT segmentation engine
    """
    # Convert OpenCV BGR to RGB and then to PIL Image
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Get original dimensions
    w, h = pil_image.size  # PIL (width, height)
    orig_size = np.array([[w, h]], dtype=np.int64)  # Format: [[width, height]]
    
    # Transform image and add batch dimension
    img_tensor = transforms(pil_image).unsqueeze(0)
    img_numpy = img_tensor.numpy()
    
    # Run inference with TensorRT segmentation engine
    labels, boxes, scores, seg_probs, seg_preds = trt_engine.infer(img_numpy, orig_size)
    
    return pil_image, labels, boxes, scores, seg_probs, seg_preds

def create_segmentation_overlay(pil_image, seg_pred, alpha=0.6):
    """Create segmentation overlay on PIL image - matching PyTorch script approach"""
    colors = get_body_part_colors()
    
    # Create colored segmentation mask
    h, w = seg_pred.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in colors.items():
        if class_id == 0:  # Skip background
            continue
        mask = seg_pred == class_id
        colored_mask[mask] = color
    
    # Convert to PIL and resize to match original image
    mask_pil = Image.fromarray(colored_mask).resize(pil_image.size, Image.NEAREST)
    
    # Create overlay using Image.blend - same as PyTorch script
    overlay = Image.blend(pil_image, mask_pil, alpha)
    
    return overlay

def draw_legend(draw, image_size):
    """Draw color legend for body parts - matching PyTorch script"""
    colors = get_body_part_colors()
    names = get_body_part_names()
    
    legend_x = image_size[0] - 150
    legend_y = 10
    
    for i, (class_id, color) in enumerate(colors.items()):
        if class_id == 0:  # Skip background
            continue
        
        y_pos = legend_y + (i-1) * 25
        
        # Draw color box
        draw.rectangle([legend_x, y_pos, legend_x + 20, y_pos + 20], 
                      fill=color, outline="white")
        
        # Draw text
        draw.text((legend_x + 25, y_pos + 2), names[class_id], fill="white")

def draw_detections_and_segmentation(pil_image, seg_pred, labels, boxes, scores, 
                                   threshold=0.5, seg_alpha=0.6, show_detection=True, show_segmentation=True):
    """Draw both detection boxes and segmentation overlay - matching PyTorch script approach"""
    
    # First create segmentation overlay if requested
    if show_segmentation:
        result_image = create_segmentation_overlay(pil_image, seg_pred, seg_alpha)
    else:
        result_image = pil_image.copy()
    
    # Then draw detection boxes if requested
    person_count = 0
    
    if show_detection and labels and boxes and scores:
        draw = ImageDraw.Draw(result_image)
        
        # Process detection results (assuming batch size 1)
        scores_per_image = scores[0] if scores[0] is not None and len(scores[0]) > 0 else []
        boxes_per_image = boxes[0] if boxes[0] is not None and len(boxes[0]) > 0 else []
        labels_per_image = labels[0] if labels[0] is not None and len(labels[0]) > 0 else []
        
        if len(scores_per_image) > 0:
            mask = scores_per_image > threshold
            filtered_boxes = boxes_per_image[mask] if len(boxes_per_image) > 0 else []
            filtered_scores = scores_per_image[mask]
            filtered_labels = labels_per_image[mask] if len(labels_per_image) > 0 else []
            
            # Draw ONLY person detections (person class ID = 0)
            for i in range(len(filtered_scores)):
                if len(filtered_labels) > 0:
                    label_id = int(filtered_labels[i].item())
                    if label_id != 0:  # Only show person detections
                        continue
                
                person_count += 1
                if len(filtered_boxes) > 0:
                    box = filtered_boxes[i].round().int().tolist()
                    score = filtered_scores[i].item()
                    
                    # Draw bounding box
                    draw.rectangle(box, outline="red", width=3)
                    
                    # Draw label
                    text = f"Person: {score:.2f}"
                    text_y_pos = box[1] - 15 if box[1] > 20 else box[1] + 2
                    draw.text((box[0] + 2, text_y_pos), text, fill="white")
    
    # Draw legend and segmentation info if showing segmentation
    if show_segmentation:
        draw = ImageDraw.Draw(result_image)
        
        # Draw legend for body parts
        draw_legend(draw, result_image.size)
        
        # Draw segmentation info
        unique_parts = np.unique(seg_pred)
        part_names = get_body_part_names()
        active_parts = [part_names[part_id] for part_id in unique_parts if part_id != 0]
        
        if active_parts:
            parts_text = f"Body parts: {', '.join(active_parts)}"
            draw.text((10, result_image.size[1] - 30), parts_text, fill="white")
    
    parts_count = len(np.unique(seg_pred)) - 1 if show_segmentation else 0  # -1 to exclude background
    
    return result_image, person_count, parts_count

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except:
        return False

def process_video_segmentation_trt(trt_engine, input_path, output_path, threshold, seg_alpha, 
                                 show_detection, show_segmentation, resize_factor, 
                                 output_fps_target, crf, preset):
    """Process video with TensorRT segmentation engine and save output"""
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is required for video output but not found.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if resize_factor != 1.0:
        output_width = int(original_width * resize_factor)
        output_height = int(original_height * resize_factor)
    else:
        output_width = original_width
        output_height = original_height
    
    # Model transforms - matching PyTorch script
    transforms_val = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    output_fps = output_fps_target if output_fps_target is not None else original_fps
    if output_fps <= 0 or np.isnan(output_fps):
        output_fps = 30

    print(f"Input: {input_path} ({original_width}x{original_height} @ {original_fps:.2f} FPS)")
    print(f"Output: {output_path} ({output_width}x{output_height} @ {output_fps:.2f} FPS)")
    print(f"Detection threshold: {threshold}, Segmentation alpha: {seg_alpha}")
    print(f"Show detection: {show_detection}, Show segmentation: {show_segmentation}")

    # FFmpeg command
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{output_width}x{output_height}', '-pix_fmt', 'bgr24',
        '-r', str(output_fps), '-i', '-', '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', '-crf', str(crf), '-preset', preset, output_path
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_idx = 0
    total_people = 0
    total_parts_detected = 0
    processing_start_time = time.time()

    try:
        while True:
            ret, frame_bgr_original = cap.read()
            if not ret:
                break

            # Resize frame if needed
            if resize_factor != 1.0:
                frame_to_process = cv2.resize(frame_bgr_original, (output_width, output_height))
            else:
                frame_to_process = frame_bgr_original
            
            # Process frame with TensorRT segmentation model
            pil_image, labels, boxes, scores, seg_probs, seg_preds = process_frame_segmentation_trt(
                trt_engine, frame_to_process, transforms_val
            )
            
            # Get segmentation prediction for current frame
            seg_pred_single = seg_preds[0] if isinstance(seg_preds, list) else seg_preds
            if isinstance(seg_pred_single, torch.Tensor):
                seg_pred_single = seg_pred_single.cpu().numpy()
            
            # Draw detections and segmentation
            result_image, person_count, parts_count = draw_detections_and_segmentation(
                pil_image, seg_pred_single, labels, boxes, scores,
                threshold=threshold, seg_alpha=seg_alpha,
                show_detection=show_detection, show_segmentation=show_segmentation
            )
            
            total_people += person_count
            total_parts_detected += parts_count

            # Convert PIL image back to BGR NumPy array for ffmpeg
            output_frame_bgr = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            
            # Ensure correct dimensions
            if output_frame_bgr.shape[1] != output_width or output_frame_bgr.shape[0] != output_height:
                output_frame_bgr = cv2.resize(output_frame_bgr, (output_width, output_height))

            # Write frame to ffmpeg
            try:
                ffmpeg_process.stdin.write(output_frame_bgr.tobytes())
            except (BrokenPipeError, IOError):
                break

            frame_idx += 1
            
            # Progress update - matching PyTorch script style
            if frame_idx % max(1, int(output_fps)) == 0 or frame_idx == 1:
                elapsed = time.time() - processing_start_time
                fps = frame_idx / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_idx) / fps if fps > 0 else 0
                
                progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                
                sys.stdout.write(
                    f"\rFrame: {frame_idx}/{total_frames} ({progress:.1f}%) | "
                    f"FPS: {fps:.1f} | ETA: {eta:.0f}s | "
                    f"People: {person_count} | Parts: {parts_count}    "
                )
                sys.stdout.flush()

    finally:
        cap.release()
        sys.stdout.write("\n")
        
        # Robust ffmpeg cleanup - matching PyTorch script approach
        if ffmpeg_process.stdin and not ffmpeg_process.stdin.closed:
            try:
                ffmpeg_process.stdin.close()
            except:
                pass
        
        try:
            ffmpeg_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            ffmpeg_process.kill()
            ffmpeg_process.wait()
        
        processing_duration = time.time() - processing_start_time
        final_fps = frame_idx / processing_duration if processing_duration > 0 else 0
        
        print(f"Processed {frame_idx} frames in {processing_duration:.2f}s ({final_fps:.2f} FPS)")
        print(f"Total people detected: {total_people}")
        if frame_idx > 0:
            print(f"Average body parts per frame: {total_parts_detected/frame_idx:.1f}")
        
        # Check results
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"✅ Output saved: {output_path} ({size_mb:.2f} MB)")
        else:
            print(f"❌ Output file not found: {output_path}")

def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
        
    if not os.path.isfile(args.engine):
        raise FileNotFoundError(f"TensorRT engine file not found: {args.engine}")
    
    # Initialize TensorRT segmentation engine
    trt_engine = TensorRTSegmentationEngine(args.engine)
    
    if args.benchmark:
        print("Benchmark mode not implemented yet")
    else:
        # Process video
        process_video_segmentation_trt(
            trt_engine=trt_engine,
            input_path=args.input,
            output_path=args.output,
            threshold=args.threshold,
            seg_alpha=args.seg_alpha,
            show_detection=args.show_detection,
            show_segmentation=args.show_segmentation,
            resize_factor=args.resize_factor,
            output_fps_target=args.fps,
            crf=args.crf,
            preset=args.preset
        )
    
    print("Processing complete.")

if __name__ == "__main__":
    main()