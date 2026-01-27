import os
import time
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import logging

from config import ModelConfig

logger = logging.getLogger(__name__)


class EngineBuilder:
    """TensorRT engine builder for DFINE segmentation models"""
    
    def __init__(self, model_config: ModelConfig):
        """
        Initialize engine builder.
        
        Args:
            model_config: Model configuration
        """
        self.model_config = model_config
        
        # TensorRT settings
        self.max_batch_size = 1
        self.workspace_size_gb = 4  # 4GB workspace for segmentation
        self.min_timing_iterations = 2
        self.avg_timing_iterations = 1
        
        # Initialize TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
        logger.info(f"TensorRT Engine Builder initialized")
        logger.info(f"TensorRT version: {trt.__version__}")
        logger.info(f"Max batch size: {self.max_batch_size}")
        logger.info(f"Workspace size: {self.workspace_size_gb}GB")
        self._check_cuda_info()
    
    def _check_cuda_info(self):
        """Check CUDA and GPU information"""
        try:
            cuda_version = cuda.get_version()
            logger.info(f"CUDA driver version: {cuda_version}")
        except Exception as e:
            logger.warning(f"Error getting CUDA version: {e}")
        
        # Check if FP16 is supported
        builder = trt.Builder(self.trt_logger)
        has_fp16 = builder.platform_has_fast_fp16
        logger.info(f"Platform has fast FP16: {has_fp16}")
    
    def build_engine(self, onnx_path: Path, engine_path: Path, precision: str = "fp16",
                     force_rebuild: bool = False) -> Path:
        """
        Build TensorRT engine from ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            precision: Engine precision (fp16, fp32, int8)
            
        Returns:
            Path to built engine
        """
        logger.info(f"Building TensorRT engine")
        logger.info(f"Input ONNX: {onnx_path}")
        logger.info(f"Output Engine: {engine_path}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Workspace: {self.workspace_size_gb}GB")
        logger.info(f"Max batch size: {self.max_batch_size}")
        
        # Ensure output directory exists
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if engine already exists
        if engine_path.exists():
            if force_rebuild:
                logger.warning(f"Engine file already exists: {engine_path}")
                logger.info("Force rebuild enabled; overwriting existing engine...")
            else:
                logger.warning(f"Engine file already exists: {engine_path}")
                logger.info("Skipping build (use --force to rebuild)")
                return engine_path
        
        start_time = time.time()
        
        logger.info("Creating TensorRT builder and config...")
        # Create builder and config
        builder = trt.Builder(self.trt_logger)
        config = builder.create_builder_config()
        
        # Set max workspace size (in bytes) - Updated for TensorRT 10.x
        workspace_bytes = self.workspace_size_gb * (1 << 30)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        
        logger.info(f"Setting precision mode: {precision}")
        # Set precision flags
        if precision == "fp16":
            if builder.platform_has_fast_fp16:
                logger.info("Enabling FP16 mode")
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                logger.warning("Platform doesn't support fast FP16, falling back to FP32")
                precision = "fp32"
        elif precision == "int8":
            logger.info("Enabling INT8 mode")
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: INT8 calibration would be needed for production use
        elif precision == "fp32":
            logger.info("Using FP32 precision")
        else:
            logger.warning(f"Unknown precision {precision}, using FP32")
            precision = "fp32"
        
        # Set optimization flags
        try:
            config.builder_optimization_level = 3  # Maximum optimization
            logger.info("Set builder optimization level to 3 (maximum)")
        except:
            logger.info("Builder optimization level not available in this TensorRT version")
        
        logger.info("Creating network definition with explicit batch...")
        # Create network definition with explicit batch flag
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        
        # Parse ONNX model
        parser = trt.OnnxParser(network, self.trt_logger)
        
        # Read the ONNX model file (use parse_from_file for external data)
        logger.info(f"Parsing ONNX segmentation model...")
        logger.info("This may take a few minutes for complex models...")
        try:
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
            
            onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(f"ONNX file size: {onnx_size_mb:.1f} MB")
            
            if not parser.parse_from_file(str(onnx_path)):
                logger.error("Failed to parse ONNX model:")
                for error in range(parser.num_errors):
                    logger.error(f"  {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")
        except Exception as e:
            logger.error(f"Error reading ONNX file: {e}")
            raise
        
        logger.info("ONNX model parsed successfully")
        
        # Print network information
        logger.info(f"Network inputs: {network.num_inputs}")
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            logger.info(f"  Input {i}: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}")
        
        logger.info(f"Network outputs: {network.num_outputs}")
        expected_outputs = ['labels', 'boxes', 'scores', 'seg_probs', 'seg_preds']
        for i in range(network.num_outputs):
            tensor = network.get_output(i)
            expected_name = expected_outputs[i] if i < len(expected_outputs) else f"output_{i}"
            logger.info(f"  Output {i}: {tensor.name} (expected: {expected_name}), shape: {tensor.shape}, dtype: {tensor.dtype}")
        
        if network.num_outputs != 5:
            logger.warning(f"Expected 5 outputs for segmentation model, found {network.num_outputs}")
            logger.warning("Expected outputs: labels, boxes, scores, seg_probs, seg_preds")
        
        # Create optimization profile for dynamic shapes
        logger.info("Creating optimization profile for dynamic input shapes...")
        profile = builder.create_optimization_profile()
        
        # Set shapes for each input tensor
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_name = input_tensor.name
            input_shape = input_tensor.shape
            
            # Check if this input has dynamic dimensions
            if -1 in input_shape:
                # Create a shape with actual values (replacing -1 with concrete values)
                min_shape = []
                opt_shape = []
                max_shape = []
                
                for dim in input_shape:
                    if dim == -1:
                        # This is a dynamic dimension, typically batch size
                        min_shape.append(1)           # Minimum batch size
                        opt_shape.append(1)           # Optimal batch size
                        max_shape.append(self.max_batch_size)  # Maximum batch size
                    else:
                        # Fixed dimension
                        min_shape.append(dim)
                        opt_shape.append(dim)
                        max_shape.append(dim)
                
                # Convert to tuple
                min_shape = tuple(min_shape)
                opt_shape = tuple(opt_shape)
                max_shape = tuple(max_shape)
                
                logger.info(f"  Setting profile for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        
        # Add the optimization profile to the config
        config.add_optimization_profile(profile)
        logger.info("Optimization profile added to config")
        
        # Build engine - Updated for TensorRT 10.x
        logger.info("Building TensorRT segmentation engine...")
        logger.info("This process may take 10-30 minutes depending on model complexity and GPU...")
        logger.info("Progress will be logged periodically...")
        
        try:
            # Log progress every 30 seconds during build
            build_start_time = time.time()
            
            plan = builder.build_serialized_network(network, config)
            if not plan:
                raise RuntimeError("Failed to build TensorRT engine!")
                
            build_time = time.time() - build_start_time
            logger.info(f"Engine built successfully in {build_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Engine building failed: {e}")
            raise RuntimeError(f"Failed to build TensorRT engine: {e}")
        
        # Save engine to file
        logger.info("Saving engine to file...")
        try:
            with open(engine_path, 'wb') as f:
                f.write(plan)
        except Exception as e:
            logger.error(f"Failed to save engine: {e}")
            raise RuntimeError(f"Failed to save engine to {engine_path}: {e}")
        
        total_time = time.time() - start_time
        file_size = engine_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Segmentation engine built successfully in {total_time:.2f} seconds")
        logger.info(f"Engine saved to: {engine_path}")
        logger.info(f"Engine file size: {file_size:.1f} MB")
        
        # Verify the engine
        logger.info("Verifying engine...")
        if not self._verify_engine(engine_path):
            raise RuntimeError("Engine verification failed!")
        
        return engine_path
    
    def _verify_engine(self, engine_path: Path) -> bool:
        """Verify the TensorRT engine by loading it"""
        logger.info(f"Verifying segmentation engine: {engine_path}")
        
        runtime = trt.Runtime(self.trt_logger)
        
        try:
            with open(engine_path, 'rb') as f:
                engine_bytes = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            if not engine:
                logger.error("Failed to deserialize engine!")
                return False
            
            logger.info("Segmentation engine verification successful")
            
            # Print engine information using TensorRT 10.x API
            try:
                # Try new API first (TensorRT 10.x)
                input_count = 0
                output_count = 0
                for i in range(engine.num_io_tensors):
                    tensor_name = engine.get_tensor_name(i)
                    if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                        input_count += 1
                    else:
                        output_count += 1
                        
                logger.info(f"Engine inputs: {input_count}")
                logger.info(f"Engine outputs: {output_count}")
                
                # Check that we have the expected number of outputs for segmentation
                if output_count == 5:
                    logger.info("Correct number of outputs for segmentation model (5)")
                else:
                    logger.warning(f"Warning: Expected 5 outputs, found {output_count}")
                
                for i in range(engine.num_io_tensors):
                    tensor_name = engine.get_tensor_name(i)
                    shape = engine.get_tensor_shape(tensor_name)
                    dtype = engine.get_tensor_dtype(tensor_name)
                    if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                        logger.info(f"  Input: {tensor_name}, shape: {shape}, dtype: {dtype}")
                        if tensor_name == "orig_target_sizes" and dtype != trt.DataType.INT64:
                            logger.warning("orig_target_sizes is not INT64; TensorRT may cast and cause issues")
                    else:
                        # Try to identify the output type
                        output_type = "unknown"
                        if 'label' in tensor_name.lower() or 'class' in tensor_name.lower():
                            output_type = "detection_labels"
                        elif 'box' in tensor_name.lower():
                            output_type = "detection_boxes"
                        elif 'score' in tensor_name.lower():
                            output_type = "detection_scores"
                        elif 'seg_prob' in tensor_name.lower():
                            output_type = "segmentation_probabilities"
                        elif 'seg_pred' in tensor_name.lower():
                            output_type = "segmentation_predictions"
                        
                        logger.info(f"  Output: {tensor_name} ({output_type}), shape: {shape}, dtype: {dtype}")
                        if tensor_name in ("labels", "seg_preds") and dtype != trt.DataType.INT64:
                            logger.warning(f"{tensor_name} is not INT64; downstream may expect int64")
            except:
                # Fall back to old API (just in case)
                logger.info("Note: Using legacy API for engine inspection")
                input_count = sum(1 for i in range(engine.num_bindings) if engine.binding_is_input(i))
                output_count = engine.num_bindings - input_count
                logger.info(f"Engine inputs: {input_count}")
                logger.info(f"Engine outputs: {output_count}")
                
                if output_count == 5:
                    logger.info("Correct number of outputs for segmentation model (5)")
                else:
                    logger.warning(f"Warning: Expected 5 outputs, found {output_count}")
                
                for i in range(engine.num_bindings):
                    name = engine.get_binding_name(i)
                    shape = engine.get_binding_shape(i)
                    dtype = engine.get_binding_dtype(i)
                    if engine.binding_is_input(i):
                        logger.info(f"  Input {i}: {name}, shape: {shape}, dtype: {dtype}")
                        if name == "orig_target_sizes" and dtype != trt.DataType.INT64:
                            logger.warning("orig_target_sizes is not INT64; TensorRT may cast and cause issues")
                    else:
                        logger.info(f"  Output {i}: {name}, shape: {shape}, dtype: {dtype}")
                        if name in ("labels", "seg_preds") and dtype != trt.DataType.INT64:
                            logger.warning(f"{name} is not INT64; downstream may expect int64")
            
            return True
        except Exception as e:
            logger.error(f"Engine verification failed: {e}")
            return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx", default="models/onnx/dfine_segmentation_x_fd256_miou73.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--engine", default="models/engines/dfine_0.73_glass_wall.engine",
                        help="Path to output engine")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"],
                        help="Engine precision")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild if engine already exists")
    args = parser.parse_args()

    engine_builder = EngineBuilder(ModelConfig())
    engine_builder.build_engine(
        onnx_path=Path(args.onnx),
        engine_path=Path(args.engine),
        precision=args.precision,
        force_rebuild=args.force
    )