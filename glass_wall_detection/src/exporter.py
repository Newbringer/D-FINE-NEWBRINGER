import os
import sys
import torch
from pathlib import Path
import logging
import warnings

# Add DFINE to path (mirror train.py behavior)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    sys.path.insert(0, project_root)

from config import ModelConfig
from model_architecture import create_segmentation_model, create_wrapper_model

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TracerWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Constant folding.*")

logger = logging.getLogger(__name__)


class DFINEExporter:
    """DFINE ONNX exporter for TensorRT conversion"""
    
    def __init__(self, model_config: ModelConfig):
        """
        Initialize exporter.
        
        Args:
            model_config: Model configuration
        """
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ONNX export settings
        self.input_shape = [1, 3, 640, 640]
        self.opset_version = 18
        
        logger.info(f"DFINE Exporter initialized")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"ONNX opset version: {self.opset_version}")
    
    def export_to_onnx(self, checkpoint_path: Path, original_weights_path: Path, 
                      config_path: Path, model_size: str = "x", num_classes: int = 81) -> Path:
        """
        Export trained model to ONNX format.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            original_weights_path: Path to original DFINE weights
            config_path: Path to DFINE config
            model_size: Model size (x, l, m, s, n)
            num_classes: Number of detection classes (80 for COCO, 81 for COCO+custom)
            
        Returns:
            Path to exported ONNX file
        """
        logger.info("="*80)
        logger.info("STARTING ONNX EXPORT PROCESS")
        logger.info("="*80)
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Original weights: {original_weights_path}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Model size: {model_size}")
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"ONNX opset: {self.opset_version}")
        
        # Load hyperparameters from checkpoint
        logger.info("\n[1/6] Loading hyperparameters from checkpoint...")
        hyperparams = self._load_hyperparameters_from_checkpoint(checkpoint_path)
        
        # Ensure num_classes is in hyperparams
        hyperparams['num_classes'] = num_classes
        logger.info(f"✓ Hyperparameters loaded (num_classes={num_classes})")
        
        # Create model with exact architecture
        logger.info("\n[2/6] Creating model architecture...")
        logger.info("This may take a few minutes for first-time model creation...")
        
        model = create_segmentation_model(
            config_path, original_weights_path, checkpoint_path, hyperparams, model_size
        )
        model = model.to(self.device)
        model.eval()
        
        logger.info("✓ Model created successfully")
        
        # Create wrapper model for ONNX export
        logger.info("\n[3/6] Creating ONNX wrapper model...")
        wrapper_model = create_wrapper_model(model, config_path, num_classes=num_classes)
        wrapper_model.eval()
        logger.info("✓ Wrapper model created")
        
        # Create dummy input
        logger.info("\n[4/6] Creating dummy inputs and testing forward pass...")
        data = torch.randn(*self.input_shape).to(self.device)
        size = torch.tensor([[self.input_shape[3], self.input_shape[2]]]).to(self.device)  # width, height
        
        # Test forward pass
        with torch.no_grad():
            try:
                test_outputs = wrapper_model(data, size)
                logger.info(f"✓ Test forward pass successful")
                logger.info(f"  Number of outputs: {len(test_outputs)} tensors")
                for i, out in enumerate(test_outputs):
                    logger.info(f"  Output {i}: shape {out.shape}, dtype {out.dtype}")
            except Exception as e:
                logger.error(f"✗ Test forward pass failed: {e}")
                raise
        
        # Define dynamic axes for ONNX export
        logger.info("\n[5/6] Setting up ONNX export configuration...")
        dynamic_axes = {
            "images": {0: "N"},  # Batch dimension
            "orig_target_sizes": {0: "N"},  # Batch dimension
            "labels": {0: "N"},
            "boxes": {0: "N"},
            "scores": {0: "N"},
            "seg_probs": {0: "N"},
            "seg_preds": {0: "N"},
        }
        
        # Determine output file name
        miou_int = int(hyperparams.get('best_miou', 0.85) * 100)
        output_filename = f"dfine_segmentation_{model_size}_fd{hyperparams.get('feature_dim', 256)}_miou{miou_int}_cls{num_classes}.onnx"
        output_path = self.model_config.cache_dir / "onnx" / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"  Output file: {output_filename}")
        logger.info(f"  Full path: {output_path}")
        
        # Export to ONNX with proper external data handling
        logger.info("\n[6/6] Exporting to ONNX format...")
        logger.info("This may take several minutes...")
        
        # Suppress ONNX export warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            try:
                # Export with external data properly configured
                torch.onnx.export(
                    wrapper_model,
                    (data, size),
                    str(output_path),
                    input_names=["images", "orig_target_sizes"],
                    output_names=["labels", "boxes", "scores", "seg_probs", "seg_preds"],
                    dynamic_axes=dynamic_axes,
                    opset_version=self.opset_version,
                    verbose=False,
                    do_constant_folding=True,
                    export_params=True
                )
                logger.info("✓ ONNX export completed successfully")
                
                # Check if external data file was created
                external_data_path = output_path.with_suffix('.onnx.data')
                if external_data_path.exists():
                    logger.info(f"✓ External data file created: {external_data_path.name}")
                    ext_size = external_data_path.stat().st_size / (1024 * 1024)
                    logger.info(f"  External data size: {ext_size:.1f} MB")
                
            except Exception as e:
                logger.error(f"✗ ONNX export failed: {e}")
                raise
        
        # Convert to single file ONNX (embed all data)
        logger.info("\nConverting to single-file ONNX (embedding external data)...")
        self._convert_to_single_file_onnx(output_path)
        
        # Verify ONNX model
        logger.info("\nVerifying ONNX model...")
        self._verify_onnx_model(output_path)
        
        # Simplify ONNX model
        logger.info("\nOptimizing ONNX model...")
        self._simplify_onnx_model(output_path)
        
        # Final summary
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info("\n" + "="*80)
        logger.info("ONNX EXPORT COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Output file: {output_path}")
        logger.info(f"File size: {file_size:.1f} MB")
        logger.info(f"Model: {model_size}, Classes: {num_classes}, mIOU: {miou_int}%")
        logger.info("="*80)
        
        return output_path
    
    def _convert_to_single_file_onnx(self, onnx_path: Path):
        """Convert ONNX with external data to single file with embedded data"""
        try:
            import onnx
            from onnx.external_data_helper import convert_model_to_external_data, convert_model_from_external_data
            
            external_data_path = onnx_path.with_suffix('.onnx.data')
            
            # Check if external data exists
            if not external_data_path.exists():
                logger.info("✓ Model already uses embedded data (no external file)")
                return
            
            logger.info("Converting external data to embedded format...")
            
            # Load model with external data
            onnx_model = onnx.load(str(onnx_path), load_external_data=True)
            
            # Save with embedded data
            temp_path = onnx_path.with_suffix('.onnx.temp')
            onnx.save(onnx_model, str(temp_path))
            
            # Replace original file
            import shutil
            shutil.move(str(temp_path), str(onnx_path))
            
            # Remove external data file
            if external_data_path.exists():
                external_data_path.unlink()
                logger.info(f"✓ Removed external data file: {external_data_path.name}")
            
            logger.info("✓ Converted to single-file ONNX with embedded data")
            
        except ImportError:
            logger.warning("⚠ ONNX package not found, skipping conversion")
            logger.info("Install with: pip install onnx")
        except Exception as e:
            logger.warning(f"⚠ Failed to convert to single file: {e}")
            logger.info("Model may still work with external data")
    
    def _load_hyperparameters_from_checkpoint(self, checkpoint_path: Path) -> dict:
        """Load hyperparameters from saved checkpoint"""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract hyperparameters
        if 'hyperparameters' in checkpoint:
            hyperparams = checkpoint['hyperparameters']
            logger.info(f"Found saved hyperparameters:")
            for key, value in hyperparams.items():
                logger.info(f"  {key}: {value}")
            return hyperparams
        else:
            logger.warning("No hyperparameters found in checkpoint, using defaults")
            return {
                'feature_dim': 256,
                'dropout_rate': 0.1,
                'best_miou': 0.85,
                'num_classes': 81
            }
    
    def _verify_onnx_model(self, onnx_path: Path):
        """Verify the exported ONNX model"""
        try:
            import onnx
            logger.info("Running ONNX model integrity check...")
            
            # Load model (this will fail if external data is missing)
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model integrity check passed")
            
            # Print model info
            logger.info(f"ONNX Model Information:")
            logger.info(f"  Inputs: {len(onnx_model.graph.input)}")
            logger.info(f"  Outputs: {len(onnx_model.graph.output)}")
            logger.info(f"  Nodes: {len(onnx_model.graph.node)}")
            logger.info(f"  Opset version: {onnx_model.opset_import[0].version}")
            
            # Check for external data
            external_data_path = onnx_path.with_suffix('.onnx.data')
            if external_data_path.exists():
                logger.warning("⚠ Model still has external data file!")
                logger.warning(f"  TensorRT needs: {onnx_path}")
                logger.warning(f"  And also needs: {external_data_path}")
            else:
                logger.info("✓ Model has all data embedded (no external file needed)")
            
        except ImportError:
            logger.warning("⚠ ONNX package not found, skipping model verification")
            logger.info("Install with: pip install onnx")
        except Exception as e:
            logger.error(f"✗ ONNX model verification failed: {e}")
            logger.warning("Model may have issues - check external data file")
    
    def _simplify_onnx_model(self, onnx_path: Path):
        """Simplify the ONNX model"""
        try:
            import onnx
            import onnxsim
            logger.info("Running ONNX model simplification...")
            
            # Create input shapes dictionary
            input_shapes = {
                "images": self.input_shape, 
                "orig_target_sizes": [self.input_shape[0], 2]
            }
            
            # Load and simplify
            onnx_model = onnx.load(str(onnx_path))
            model_simp, check = onnxsim.simplify(
                onnx_model, 
                test_input_shapes=input_shapes,
            )
            
            if check:
                onnx.save(model_simp, str(onnx_path))
                logger.info(f"✓ ONNX model simplified and saved")
            else:
                logger.warning("⚠ ONNX model simplification could not be validated")
                logger.info("Model is functional but not optimized")
                
        except ImportError:
            logger.warning("⚠ onnx-simplifier package not found, skipping optimization")
            logger.info("Install with: pip install onnx-simplifier")
        except Exception as e:
            logger.warning(f"⚠ ONNX model simplification failed: {e}")
            logger.info("Model is functional but not optimized")


if __name__ == "__main__":
    exporter = DFINEExporter(ModelConfig())
    exporter.export_to_onnx(
        checkpoint_path=Path("../outputs/glass_detection_segmentation/dfine_0.73_glass_wall.pth"),
        original_weights_path=Path("../models/dfine.pth"),
        config_path=Path("../models/dfine_hgnetv2_x_obj2coco.yml"),
        model_size="x",
        num_classes=81
    )