"""Tests for LoRA config."""
import pytest
from src.lora_config import LoraConfigParams, LoraConfigBuilder, get_peft_config, HAS_PEFT, count_trainable_params


class TestLoraConfigBuilder:
    def test_default_config(self):
        builder = LoraConfigBuilder()
        config = builder.build_dict()
        assert config["r"] == 8
        assert config["lora_alpha"] == 16

    def test_custom_r(self):
        builder = LoraConfigBuilder().with_r(32)
        config = builder.build_dict()
        assert config["r"] == 32

    def test_chained_builders(self):
        builder = LoraConfigBuilder().with_r(16).with_alpha(32).with_dropout(0.05)
        config = builder.build_dict()
        assert config["r"] == 16
        assert config["lora_alpha"] == 32
        assert config["lora_dropout"] == 0.05

    def test_with_target_modules(self):
        builder = LoraConfigBuilder().with_target_modules(["q_proj", "v_proj"])
        config = builder.build_dict()
        assert config["target_modules"] == ["q_proj", "v_proj"]

    @pytest.mark.skipif(not HAS_PEFT, reason="PEFT not installed")
    def test_build_peft_config(self):
        builder = LoraConfigBuilder().with_r(8)
        config = builder.build_peft_config()
        assert config.r == 8


class TestGetPeftConfig:
    def test_defaults(self):
        config = get_peft_config()
        assert config["r"] == 8
        assert config["lora_dropout"] == 0.1

    def test_overrides(self):
        config = get_peft_config(r=16, lora_alpha=32)
        assert config["r"] == 16
        assert config["lora_alpha"] == 32


class TestCountTrainableParams:
    def test_returns_dict(self):
        # Test with a simple mock model
        class MockModel:
            def parameters(self):
                class P:
                    numel = lambda self: 1000
                    requires_grad = True
                return [P(), P(), P(), P(), P()]  # 5 * 1000 = 5000
        result = count_trainable_params(MockModel())
        assert "trainable" in result
        assert "total" in result
        assert "trainable_percent" in result
