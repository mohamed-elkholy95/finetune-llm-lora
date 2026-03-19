"""LoRA configuration factory."""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.config import LORA_CONFIG

logger = logging.getLogger(__name__)

HAS_PEFT = False
try:
    from peft import LoraConfig, TaskType, get_peft_model
    HAS_PEFT = True
except ImportError:
    logger.info("PEFT not available")


@dataclass
class LoraConfigParams:
    """LoRA configuration parameters."""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules: Optional[list] = None
    task_type: str = "CAUSAL_LM"


class LoraConfigBuilder:
    """Builder for LoRA configurations."""

    def __init__(self, params: Optional[LoraConfigParams] = None) -> None:
        self._params = params or LoraConfigParams()

    def with_r(self, r: int) -> "LoraConfigBuilder":
        self._params.r = r
        return self

    def with_alpha(self, alpha: int) -> "LoraConfigBuilder":
        self._params.lora_alpha = alpha
        return self

    def with_dropout(self, dropout: float) -> "LoraConfigBuilder":
        self._params.lora_dropout = dropout
        return self

    def with_target_modules(self, modules: list) -> "LoraConfigBuilder":
        self._params.target_modules = modules
        return self

    def build_dict(self) -> Dict[str, Any]:
        """Build config as dictionary."""
        return {
            "r": self._params.r,
            "lora_alpha": self._params.lora_alpha,
            "lora_dropout": self._params.lora_dropout,
            "bias": self._params.bias,
            "target_modules": self._params.target_modules,
        }

    def build_peft_config(self):
        """Build actual PEFT LoraConfig."""
        if not HAS_PEFT:
            logger.warning("PEFT not available — returning dict config")
            return self.build_dict()

        return LoraConfig(
            r=self._params.r,
            lora_alpha=self._params.lora_alpha,
            lora_dropout=self._params.lora_dropout,
            bias=self._params.bias,
            target_modules=self._params.target_modules,
            task_type=TaskType.CAUSAL_LM,
        )


def get_peft_config(**overrides) -> Dict[str, Any]:
    """Get LoRA config with optional overrides.

    Args:
        **overrides: Override default config values.

    Returns:
        Config dictionary.
    """
    config = {**LORA_CONFIG, **overrides}
    logger.info("LoRA config: r=%d, alpha=%d, dropout=%.2f", config["r"], config["lora_alpha"], config["lora_dropout"])
    return config


def count_trainable_params(model) -> Dict[str, int]:
    """Count trainable vs total parameters.

    Args:
        model: HuggingFace model.

    Returns:
        Dict with trainable, total, and percentage.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_percent": round(100 * trainable / total, 2) if total > 0 else 0,
    }
