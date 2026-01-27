from dataclasses import dataclass
from loguru import logger
import pathlib
from typing import Dict, Any, Optional, Tuple
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class S3Config(BaseSettings):
    endpoint_url: Optional[str] = "https://ams3.digitaloceanspaces.com"
    access_key_id: Optional[str] = "DO00RT3WCHNFZZGBNEMK"
    secret_access_key: Optional[str] = "KpV2awirASOGe2PXlX/sXdBPeoeektZqXQ9jVZDQSe4"
    bucket_name: Optional[str] = "newbringer"
    directory: str = "newbringer/dfine-models"
    region_name: str = "ams3"

    class Config:
        env_prefix = "S3_"


class ModelConfig(BaseSettings):
    name: str = "dfine_segmentation"
    device_type: str = Field(default="cuda")
    cache_dir: pathlib.Path = Field(default=pathlib.Path("models"))
    input_size: Tuple[int, int] = Field(default=(640, 640))
    confidence_threshold: float = Field(default=0.5)
    verbose: bool = Field(default=False)
    engine_pool_size: int = Field(default=10)  # Number of parallel engines
    crosshair_region_size: int = Field(default=3)
    
    # TensorRT specific settings
    engine_precision: str = Field(default="fp16")  # fp32, fp16, int8
    use_int8: bool = Field(default=False)
    engine_suffix: str = Field(default="")  # Optional suffix for engine files
    
    @field_validator("device_type")
    @classmethod
    def validate_device(cls, v: str) -> str:
        import torch
        if v == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        return v

    class Config:
        env_prefix = "DFINE_MODEL_"


class AppConfig(BaseSettings):
    model: ModelConfig = Field(default_factory=ModelConfig)
    s3: S3Config = Field(default_factory=S3Config)

    class Config:
        env_nested_delimiter = "__"


@dataclass
class DFINEResult:
    labels: Any
    boxes: Any  
    scores: Any
    seg_probs: Any
    seg_preds: Any
    persons: list
    visualization: Any
    processing_stats: Dict[str, float]