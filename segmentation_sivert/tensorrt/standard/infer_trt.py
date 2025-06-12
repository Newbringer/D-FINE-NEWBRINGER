#!/usr/bin/env python3
"""
DFINE Segmentation TensorRT Inference and Benchmark Script

Clean, focused script for your segmentation model with two main modes:
1. Video processing with segmentation visualization 
2. Pure inference benchmarking with parallel streams

Usage:
    # Process video with segmentation visualization
    python infer_segmentation_trt.py -e model.engine -i input.mp4 -o output.mp4
    
    # Pure inference benchmark (single stream)
    python infer_segmentation_trt.py -e model.engine -i input.mp4 --benchmark
    
    # Parallel inference benchmark (multiple streams)
    python infer_segmentation_trt.py -e model.engine -i input.mp4 --benchmark --parallel-streams 4
"""

import os
import sys
import argparse
import time
import statistics
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import deque
import psutil
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import concurrent.futures
from queue import Queue
import multiprocessing as mp
import subprocess
import tempfile

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    total_time: float
    inference_time: float
    pure_inference_fps: float
    frames_processed: int
    memory_usage_mb: float
    parallel_streams: int = 1
    total_throughput_fps: float = 0.0

class TimingProfiler:
    """High-precision timing profiler with CUDA synchronization"""
    
    def __init__(self):
        self.timings = {}
        self.cuda_events = {}
        self.current_stream = cuda.Stream()
        
    def create_cuda_events(self, name: str):
        """Create CUDA events for precise GPU timing"""
        self.cuda_events[f"{name}_start"] = cuda.Event()
        self.cuda_events[f"{name}_end"] = cuda.Event()
    
    def start_cuda_timing(self, name: str):
        """Start CUDA timing"""
        if f"{name}_start" not in self.cuda_events:
            self.create_cuda_events(name)
        self.cuda_events[f"{name}_start"].record(self.current_stream)
    
    def end_cuda_timing(self, name: str):
        """End CUDA timing and return duration in seconds"""
        self.cuda_events[f"{name}_end"].record(self.current_stream)
        self.cuda_events[f"{name}_end"].synchronize()
        
        duration_ms = self.cuda_events[f"{name}_start"].time_till(self.cuda_events[f"{name}_end"])
        duration_s = duration_ms / 1000.0
        
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration_s)
        
        return duration_s

class TensorRTEngine:
    """Optimized TensorRT engine wrapper"""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = None
        self.engine = None
        self.context = None
        self.stream = None
        
        # Memory management
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.input_shapes = {}
        self.output_shapes = {}
        
        self._load_engine()
        self._allocate_memory()
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        print(f"🔧 Loading TensorRT engine: {self.engine_path}")
        
        self.runtime = trt.Runtime(self.logger)
        
        with open(self.engine_path, 'rb') as f:
            engine_bytes = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if not self.engine:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
        if not self.context:
            raise RuntimeError("Failed to create TensorRT execution context")
        
        self.stream = cuda.Stream()
        print(f"✅ Engine loaded successfully")
    
    def _allocate_memory(self):
        """Pre-allocate host and device memory"""
        print(f"🧠 Allocating memory buffers...")
        
        # Check TensorRT version
        use_new_api = hasattr(self.engine, 'num_io_tensors')
        
        if use_new_api:
            # TensorRT 10.x+ API
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(tensor_name)
                dtype = self.engine.get_tensor_dtype(tensor_name)
                is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
                
                # Handle dynamic shapes - set to batch size 1
                if -1 in shape:
                    actual_shape = tuple(1 if dim == -1 else dim for dim in shape)
                    self.context.set_input_shape(tensor_name, actual_shape)
                    shape = actual_shape
                
                # Convert dtype and calculate size
                if dtype == trt.DataType.FLOAT:
                    np_dtype, element_size = np.float32, 4
                elif dtype == trt.DataType.HALF:
                    np_dtype, element_size = np.float16, 2
                elif dtype == trt.DataType.INT32:
                    np_dtype, element_size = np.int32, 4
                elif dtype == trt.DataType.INT64:
                    np_dtype, element_size = np.int64, 8
                else:
                    np_dtype, element_size = np.float32, 4
                
                size = trt.volume(shape)
                
                # Allocate memory
                host_mem = cuda.pagelocked_empty(size, np_dtype)
                device_mem = cuda.mem_alloc(size * element_size)
                
                if is_input:
                    self.host_inputs.append(host_mem)
                    self.device_inputs.append(device_mem)
                    self.input_shapes[tensor_name] = shape
                else:
                    self.host_outputs.append(host_mem)
                    self.device_outputs.append(device_mem)
                    self.output_shapes[tensor_name] = shape
                
                # Set tensor address
                self.context.set_tensor_address(tensor_name, int(device_mem))
                
                print(f"   {'📥' if is_input else '📤'} {tensor_name}: {shape} ({size * element_size / 1024 / 1024:.1f} MB)")
        
        else:
            # Legacy TensorRT API
            self.bindings = []
            for i in range(self.engine.num_bindings):
                binding_name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                dtype = self.engine.get_binding_dtype(i)
                is_input = self.engine.binding_is_input(i)
                
                if -1 in shape:
                    actual_shape = tuple(1 if dim == -1 else dim for dim in shape)
                    self.context.set_binding_shape(i, actual_shape)
                    shape = actual_shape
                
                if dtype == trt.DataType.FLOAT:
                    np_dtype, element_size = np.float32, 4
                elif dtype == trt.DataType.INT64:
                    np_dtype, element_size = np.int64, 8
                else:
                    np_dtype, element_size = np.float32, 4
                
                size = trt.volume(shape)
                host_mem = cuda.pagelocked_empty(size, np_dtype)
                device_mem = cuda.mem_alloc(size * element_size)
                
                if is_input:
                    self.host_inputs.append(host_mem)
                    self.device_inputs.append(device_mem)
                    self.input_shapes[binding_name] = shape
                else:
                    self.host_outputs.append(host_mem)
                    self.device_outputs.append(device_mem)
                    self.output_shapes[binding_name] = shape
                
                self.bindings.append(int(device_mem))
    
    def infer(self, inputs: List[np.ndarray], profiler: TimingProfiler) -> List[np.ndarray]:
        """Perform inference with timing"""
        
        # Copy inputs to device
        profiler.start_cuda_timing("memory_h2d")
        for i, input_data in enumerate(inputs):
            np.copyto(self.host_inputs[i][:input_data.size], input_data.ravel())
            cuda.memcpy_htod_async(self.device_inputs[i], self.host_inputs[i], self.stream)
        profiler.end_cuda_timing("memory_h2d")
        
        # Execute inference
        profiler.start_cuda_timing("inference")
        if hasattr(self.context, 'execute_async_v3'):
            success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        elif hasattr(self.context, 'execute_async_v2'):
            success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            success = self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        
        if not success:
            raise RuntimeError("TensorRT inference failed")
        profiler.end_cuda_timing("inference")
        
        # Copy outputs from device
        profiler.start_cuda_timing("memory_d2h")
        outputs = []
        for i, (host_output, device_output, shape) in enumerate(
            zip(self.host_outputs, self.device_outputs, self.output_shapes.values())
        ):
            cuda.memcpy_dtoh_async(host_output, device_output, self.stream)
            self.stream.synchronize()
            output = host_output[:np.prod(shape)].reshape(shape).copy()
            outputs.append(output)
        profiler.end_cuda_timing("memory_d2h")
        
        return outputs

def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess single frame for inference"""
    # Resize frame
    resized = cv2.resize(frame, target_size)
    
    # Convert to RGB and normalize
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb_frame.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    
    # Convert to NCHW format
    input_tensor = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
    
    # Target size tensor
    target_size_tensor = np.array([[target_size[0], target_size[1]]], dtype=np.int64)
    
    return input_tensor, target_size_tensor

def preprocess_entire_video(input_video: str, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
    """Preprocess entire video for benchmarking"""
    
    print(f"🎬 Preprocessing entire video for benchmark mode...")
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_video}")
    
    # Get video info
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    video_info = {
        'fps': video_fps,
        'total_frames': total_frames,
        'width': width,
        'height': height
    }
    
    print(f"📊 Processing {total_frames:,} frames ({width}x{height} @ {video_fps:.1f} FPS)")
    
    preprocessed_frames = []
    target_size_tensors = []
    
    frame_count = 0
    start_time = time.perf_counter()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
            
            input_tensor, target_size_tensor = preprocess_frame(frame)
            preprocessed_frames.append(input_tensor)
            target_size_tensors.append(target_size_tensor)
            
            frame_count += 1
            
            # Progress reporting
            if frame_count % max(1, total_frames // 20) == 0:
                elapsed = time.perf_counter() - start_time
                fps = frame_count / elapsed
                eta = (total_frames - frame_count) / fps if fps > 0 else 0
                print(f"   Progress: {frame_count:4d}/{total_frames} ({frame_count/total_frames*100:5.1f}%) | "
                      f"Speed: {fps:6.1f} FPS | ETA: {eta:4.1f}s")
    
    finally:
        cap.release()
    
    preprocessing_time = time.perf_counter() - start_time
    memory_usage_mb = sum(frame.nbytes for frame in preprocessed_frames) / 1024 / 1024
    
    print(f"✅ Preprocessing completed:")
    print(f"   📊 {frame_count:,} frames in {preprocessing_time:.2f}s ({frame_count/preprocessing_time:.1f} FPS)")
    print(f"   💾 Memory usage: {memory_usage_mb:.1f} MB")
    
    return preprocessed_frames, target_size_tensors, video_info

def create_segmentation_overlay(frame: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Create segmentation overlay on frame"""
    
    # Pascal Person Parts color map (7 classes)
    colors = [
        [0, 0, 0],         # 0: background - black
        [128, 0, 0],       # 1: head - dark red
        [255, 0, 0],       # 2: torso - red  
        [0, 128, 0],       # 3: upper arms - dark green
        [0, 255, 0],       # 4: lower arms - green
        [0, 0, 128],       # 5: upper legs - dark blue
        [0, 0, 255],       # 6: lower legs - blue
    ]
    
    height, width = frame.shape[:2]
    
    # Resize segmentation mask to frame size
    seg_resized = cv2.resize(seg_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask = (seg_resized == class_id)
        overlay[mask] = color
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
    
    return result

def draw_detections(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, 
                   confidence_threshold: float = 0.5) -> np.ndarray:
    """Draw detection boxes on frame"""
    
    height, width = frame.shape[:2]
    result = frame.copy()
    
    # COCO class names (simplified)
    class_names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light'
    }
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
              (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0)]
    
    for i in range(len(scores)):
        if scores[i] < confidence_threshold:
            continue
            
        # Convert normalized coordinates to pixel coordinates
        x1 = int(boxes[i][0] * width)
        y1 = int(boxes[i][1] * height)
        x2 = int(boxes[i][2] * width)
        y2 = int(boxes[i][3] * height)
        
        label = int(labels[i])
        color = colors[label % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        class_name = class_names.get(label, f'Class {label}')
        label_text = f'{class_name}: {scores[i]:.2f}'
        
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(result, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return result

def run_single_stream_benchmark(engine_path: str, frames_chunk: List, stream_id: int, results_queue: Queue):
    """Run benchmark on a single stream"""
    
    try:
        # Initialize CUDA for this thread
        cuda.init()
        device = cuda.Device(0)
        cuda_context = device.make_context()
        
        try:
            # Create engine instance
            engine = TensorRTEngine(engine_path)
            profiler = TimingProfiler()
            
            # Warmup
            for _ in range(3):
                inputs = [frames_chunk[0][0], frames_chunk[0][1]]
                _ = engine.infer(inputs, profiler)
            
            # Benchmark
            start_time = time.perf_counter()
            
            for input_tensor, target_size_tensor in frames_chunk:
                inputs = [input_tensor, target_size_tensor]
                outputs = engine.infer(inputs, profiler)
            
            total_time = time.perf_counter() - start_time
            fps = len(frames_chunk) / total_time
            avg_inference_time = statistics.mean(profiler.timings.get("inference", [0]))
            
            results_queue.put({
                'stream_id': stream_id,
                'frames_processed': len(frames_chunk),
                'total_time': total_time,
                'fps': fps,
                'avg_inference_time': avg_inference_time,
                'success': True
            })
            
        finally:
            cuda_context.pop()
            
    except Exception as e:
        results_queue.put({
            'stream_id': stream_id,
            'error': str(e),
            'success': False
        })

def run_parallel_benchmark(engine_path: str, preprocessed_frames: List, target_size_tensors: List, 
                          num_streams: int) -> BenchmarkResult:
    """Run parallel inference benchmark"""
    
    print(f"\n🚀 Running parallel inference benchmark with {num_streams} streams")
    print(f"📊 Total frames: {len(preprocessed_frames):,}")
    
    # Split frames across streams
    frames_per_stream = len(preprocessed_frames) // num_streams
    frame_chunks = []
    
    for i in range(num_streams):
        start_idx = i * frames_per_stream
        if i == num_streams - 1:  # Last stream gets remaining frames
            end_idx = len(preprocessed_frames)
        else:
            end_idx = (i + 1) * frames_per_stream
        
        chunk = list(zip(preprocessed_frames[start_idx:end_idx], 
                        target_size_tensors[start_idx:end_idx]))
        frame_chunks.append(chunk)
        print(f"   Stream {i+1}: {len(chunk)} frames")
    
    # Run parallel benchmark
    results_queue = Queue()
    start_time = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_streams) as executor:
        futures = []
        for i, chunk in enumerate(frame_chunks):
            future = executor.submit(run_single_stream_benchmark, engine_path, chunk, i, results_queue)
            futures.append(future)
        
        concurrent.futures.wait(futures)
    
    total_time = time.perf_counter() - start_time
    
    # Collect results
    stream_results = []
    total_frames_processed = 0
    
    while not results_queue.empty():
        result = results_queue.get()
        if result['success']:
            stream_results.append(result)
            total_frames_processed += result['frames_processed']
        else:
            print(f"❌ Stream {result['stream_id']} failed: {result['error']}")
    
    if not stream_results:
        raise RuntimeError("All streams failed")
    
    # Calculate statistics
    total_throughput_fps = total_frames_processed / total_time
    avg_per_stream_fps = statistics.mean([r['fps'] for r in stream_results])
    avg_inference_time = statistics.mean([r['avg_inference_time'] for r in stream_results])
    
    print(f"\n🎉 Parallel benchmark completed!")
    print(f"   Successful streams: {len(stream_results)}/{num_streams}")
    print(f"   Total throughput: {total_throughput_fps:.1f} FPS")
    print(f"   Average per stream: {avg_per_stream_fps:.1f} FPS")
    print(f"   Average inference time: {avg_inference_time*1000:.2f}ms")
    
    if len(stream_results) > 1:
        parallel_efficiency = total_throughput_fps / (avg_per_stream_fps * len(stream_results))
        print(f"   Parallel efficiency: {parallel_efficiency * 100:.1f}%")
    
    return BenchmarkResult(
        total_time=total_time,
        inference_time=avg_inference_time,
        pure_inference_fps=avg_per_stream_fps,
        frames_processed=total_frames_processed,
        memory_usage_mb=0.0,
        parallel_streams=len(stream_results),
        total_throughput_fps=total_throughput_fps
    )

def run_single_stream_benchmark_main(engine_path: str, preprocessed_frames: List, 
                                   target_size_tensors: List) -> BenchmarkResult:
    """Run single stream benchmark"""
    
    print(f"\n🚀 Running single stream inference benchmark")
    print(f"📊 Total frames: {len(preprocessed_frames):,}")
    
    # Initialize engine
    engine = TensorRTEngine(engine_path)
    profiler = TimingProfiler()
    
    # Warmup
    print("🔥 Warming up...")
    for _ in range(10):
        inputs = [preprocessed_frames[0], target_size_tensors[0]]
        _ = engine.infer(inputs, profiler)
    
    # Memory monitoring
    process = psutil.Process()
    initial_memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Pure inference benchmark
    print("🏃 Running pure inference benchmark...")
    start_time = time.perf_counter()
    
    for i, (input_tensor, target_size_tensor) in enumerate(zip(preprocessed_frames, target_size_tensors)):
        inputs = [input_tensor, target_size_tensor]
        outputs = engine.infer(inputs, profiler)
        
        # Progress reporting
        if (i + 1) % max(1, len(preprocessed_frames) // 20) == 0:
            elapsed = time.perf_counter() - start_time
            current_fps = (i + 1) / elapsed
            eta = (len(preprocessed_frames) - i - 1) / current_fps if current_fps > 0 else 0
            print(f"   Progress: {i + 1:4d}/{len(preprocessed_frames)} | "
                  f"FPS: {current_fps:6.1f} | ETA: {eta:4.1f}s")
    
    total_time = time.perf_counter() - start_time
    final_memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Calculate results
    pure_inference_fps = len(preprocessed_frames) / total_time
    avg_inference_time = statistics.mean(profiler.timings.get("inference", [0]))
    memory_usage_mb = final_memory_mb - initial_memory_mb
    
    print(f"\n🎉 Single stream benchmark completed!")
    print(f"   Pure inference FPS: {pure_inference_fps:.1f}")
    print(f"   Average inference time: {avg_inference_time*1000:.2f}ms")
    print(f"   Memory usage: +{memory_usage_mb:.1f} MB")
    
    return BenchmarkResult(
        total_time=total_time,
        inference_time=avg_inference_time,
        pure_inference_fps=pure_inference_fps,
        frames_processed=len(preprocessed_frames),
        memory_usage_mb=memory_usage_mb,
        parallel_streams=1,
        total_throughput_fps=pure_inference_fps
    )

def process_video_with_segmentation(engine_path: str, input_video: str, output_video: str, 
                                  max_frames: Optional[int] = None) -> BenchmarkResult:
    """Process video with segmentation visualization"""
    
    print(f"\n🎬 Processing video with segmentation")
    print(f"📥 Input: {input_video}")
    print(f"📤 Output: {output_video}")
    
    # Initialize engine
    engine = TensorRTEngine(engine_path)
    profiler = TimingProfiler()
    
    # Warmup
    print("🔥 Warming up engine...")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    input_tensor, target_size_tensor = preprocess_frame(dummy_frame)
    for _ in range(5):
        _ = engine.infer([input_tensor, target_size_tensor], profiler)
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_video}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"📊 Video: {width}x{height} @ {video_fps:.1f} FPS, processing {total_frames} frames")
    
    # Prepare temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(temp_dir, 'frames')
    os.makedirs(frames_dir)
    
    # Performance monitoring
    process = psutil.Process()
    initial_memory_mb = process.memory_info().rss / 1024 / 1024
    
    frame_count = 0
    start_time = time.perf_counter()
    fps_history = deque(maxlen=30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
            
            frame_start_time = time.perf_counter()
            
            # Preprocess frame
            input_tensor, target_size_tensor = preprocess_frame(frame)
            
            # Inference
            inputs = [input_tensor, target_size_tensor]
            outputs = engine.infer(inputs, profiler)
            
            # Extract outputs (expected: labels, boxes, scores, seg_probs, seg_preds)
            labels = outputs[0].flatten()      # Detection labels
            boxes = outputs[1][0]              # Detection boxes (N, 4)
            scores = outputs[2].flatten()      # Detection scores  
            seg_probs = outputs[3][0]          # Segmentation probabilities (7, 640, 640)
            seg_preds = outputs[4][0]          # Segmentation predictions (640, 640)
            
            # Create visualization
            vis_frame = frame.copy()
            
            # Add segmentation overlay
            vis_frame = create_segmentation_overlay(vis_frame, seg_preds.astype(np.uint8), alpha=0.6)
            
            # Add detection boxes
            vis_frame = draw_detections(vis_frame, boxes, scores, labels, confidence_threshold=0.5)
            
            # Add performance info
            frame_time = time.perf_counter() - frame_start_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(current_fps)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # Performance overlay
            perf_text = [
                f"Frame: {frame_count+1}/{total_frames}",
                f"FPS: {avg_fps:.1f}",
                f"Inference: {profiler.timings['inference'][-1]*1000:.1f}ms" if profiler.timings.get('inference') else "Inference: --ms"
            ]
            
            y_offset = 30
            for text in perf_text:
                cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # Save frame
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
            cv2.imwrite(frame_path, vis_frame)
            
            frame_count += 1
            
            # Progress reporting
            if frame_count % max(1, total_frames // 20) == 0:
                elapsed = time.perf_counter() - start_time
                processing_fps = frame_count / elapsed
                eta = (total_frames - frame_count) / processing_fps if processing_fps > 0 else 0
                print(f"   Processing: {frame_count:4d}/{total_frames} ({frame_count/total_frames*100:5.1f}%) | "
                      f"Speed: {processing_fps:6.1f} FPS | ETA: {eta:4.1f}s")
    
    finally:
        cap.release()
    
    total_time = time.perf_counter() - start_time
    final_memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Create video using ffmpeg
    print(f"🎬 Creating output video with ffmpeg...")
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Overwrite output file
        '-framerate', str(video_fps),
        '-i', os.path.join(frames_dir, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',  # High quality
        output_video
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"✅ Video saved to: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        raise
    finally:
        # Cleanup temporary files
        import shutil
        shutil.rmtree(temp_dir)
    
    # Calculate results
    overall_fps = frame_count / total_time
    avg_inference_time = statistics.mean(profiler.timings.get("inference", [0]))
    memory_usage_mb = final_memory_mb - initial_memory_mb
    
    print(f"\n🎉 Video processing completed!")
    print(f"📊 Processed {frame_count:,} frames in {total_time:.2f}s ({overall_fps:.1f} FPS)")
    print(f"⚡ Average inference time: {avg_inference_time*1000:.2f}ms")
    print(f"💾 Memory usage: +{memory_usage_mb:.1f} MB")
    print(f"🎯 Real-time factor: {overall_fps / video_fps:.2f}x")
    
    return BenchmarkResult(
        total_time=total_time,
        inference_time=avg_inference_time,
        pure_inference_fps=1.0 / avg_inference_time if avg_inference_time > 0 else 0,
        frames_processed=frame_count,
        memory_usage_mb=memory_usage_mb,
        parallel_streams=1,
        total_throughput_fps=overall_fps
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DFINE Segmentation TensorRT Inference')
    parser.add_argument('-e', '--engine', type=str, required=True,
                        help='Path to TensorRT engine file')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('-o', '--output', type=str,
                        help='Path to output video file (required for video processing mode)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run pure inference benchmark mode')
    parser.add_argument('--parallel-streams', type=int, default=10,
                        help='Number of parallel streams for benchmark (default: 1)')
    parser.add_argument('--max-frames', type=int,
                        help='Maximum number of frames to process')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Validate arguments
    if not args.benchmark and not args.output:
        print("❌ Error: --output is required when not in benchmark mode")
        sys.exit(1)
    
    if args.parallel_streams > 1 and not args.benchmark:
        print("❌ Error: --parallel-streams can only be used with --benchmark")
        sys.exit(1)
    
    # Check files exist
    if not os.path.isfile(args.engine):
        print(f"❌ Error: Engine file not found: {args.engine}")
        sys.exit(1)
    
    if not os.path.isfile(args.input):
        print(f"❌ Error: Input video not found: {args.input}")
        sys.exit(1)
    
    print("=" * 80)
    print("🚀 DFINE Segmentation TensorRT Inference")
    print("=" * 80)
    print(f"📁 Engine: {args.engine}")
    print(f"📁 Input: {args.input}")
    
    if args.benchmark:
        if args.parallel_streams > 1:
            print(f"⚡ Mode: PARALLEL BENCHMARK ({args.parallel_streams} streams)")
        else:
            print(f"⚡ Mode: SINGLE STREAM BENCHMARK")
    else:
        print(f"📁 Output: {args.output}")
        print(f"🎬 Mode: VIDEO PROCESSING")
    
    if args.max_frames:
        print(f"🔢 Max frames: {args.max_frames:,}")
    
    print("=" * 80)
    
    # System info
    print(f"💻 System: {psutil.cpu_count()} CPU cores, {psutil.virtual_memory().available / 1024**3:.1f} GB RAM")
    print(f"🎮 CUDA devices: {cuda.Device.count()}")
    if cuda.Device.count() > 0:
        device = cuda.Device(0)
        print(f"   GPU: {device.name()} ({device.total_memory() / 1024**3:.1f} GB)")
    
    try:
        if args.benchmark:
            # Benchmark mode
            preprocessed_frames, target_size_tensors, video_info = preprocess_entire_video(
                args.input, args.max_frames
            )
            
            if args.parallel_streams > 1:
                result = run_parallel_benchmark(
                    args.engine, preprocessed_frames, target_size_tensors, args.parallel_streams
                )
            else:
                result = run_single_stream_benchmark_main(
                    args.engine, preprocessed_frames, target_size_tensors
                )
        else:
            # Video processing mode
            result = process_video_with_segmentation(
                args.engine, args.input, args.output, args.max_frames
            )
        
        # Print final summary
        print(f"\n🎯 FINAL RESULTS:")
        if args.benchmark:
            if args.parallel_streams > 1:
                print(f"   🚀 PARALLEL BENCHMARK RESULTS:")
                print(f"      Streams: {result.parallel_streams}/{args.parallel_streams} successful")
                print(f"      Total throughput: {result.total_throughput_fps:.1f} FPS")
                print(f"      Average per stream: {result.pure_inference_fps:.1f} FPS")
                parallel_efficiency = result.total_throughput_fps / (result.pure_inference_fps * result.parallel_streams) * 100
                print(f"      Parallel efficiency: {parallel_efficiency:.1f}%")
                print(f"      Average inference time: {result.inference_time*1000:.2f}ms")
            else:
                print(f"   🚀 SINGLE STREAM BENCHMARK RESULTS:")
                print(f"      Pure inference FPS: {result.pure_inference_fps:.1f}")
                print(f"      Average inference time: {result.inference_time*1000:.2f}ms")
            
            print(f"   📊 Frames processed: {result.frames_processed:,}")
            print(f"   ⏱️  Total time: {result.total_time:.2f}s")
            print(f"   💾 Memory usage: +{result.memory_usage_mb:.1f} MB")
            
        else:
            print(f"   🎬 VIDEO PROCESSING RESULTS:")
            print(f"      Processing FPS: {result.total_throughput_fps:.1f}")
            print(f"      Pure inference FPS: {result.pure_inference_fps:.1f}")
            print(f"      Real-time capable: {'✅ Yes' if result.total_throughput_fps >= 30 else '⚠️ No'}")
            print(f"   📊 Frames processed: {result.frames_processed:,}")
            print(f"   ⏱️  Total time: {result.total_time:.2f}s")
            print(f"   💾 Memory usage: +{result.memory_usage_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n🎉 Processing completed successfully!")

if __name__ == "__main__":
    main()