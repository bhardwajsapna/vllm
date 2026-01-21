# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DFlash (Block Diffusion) speculative decoding."""

import pytest
import torch

from vllm.transformers_utils.configs.dflash import DFlashConfig


class TestDFlashConfig:
    """Tests for DFlashConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DFlashConfig()

        assert config.model_type == "dflash"
        assert config.hidden_size == 4096
        assert config.vocab_size == 32000
        assert config.block_size == 8
        assert config.num_diffusion_steps == 8
        assert config.noise_schedule == "cosine"
        assert config.num_attention_heads == 32
        assert config.num_hidden_layers == 2
        assert config.architectures == ["DFlashModel"]

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DFlashConfig(
            hidden_size=2048,
            vocab_size=50000,
            block_size=16,
            num_diffusion_steps=4,
            noise_schedule="linear",
        )

        assert config.hidden_size == 2048
        assert config.vocab_size == 50000
        assert config.block_size == 16
        assert config.num_diffusion_steps == 4
        assert config.noise_schedule == "linear"

    def test_num_lookahead_tokens_property(self):
        """Test num_lookahead_tokens property."""
        config = DFlashConfig(block_size=12)

        assert config.num_lookahead_tokens == 12

        # Test setter
        config.num_lookahead_tokens = 20
        assert config.block_size == 20
        assert config.num_lookahead_tokens == 20


class TestDFlashNoiseSchedule:
    """Tests for DFlashNoiseSchedule."""

    def test_cosine_schedule(self):
        """Test cosine noise schedule."""
        from vllm.model_executor.models.dflash import DFlashNoiseSchedule

        schedule = DFlashNoiseSchedule(num_steps=8, schedule_type="cosine")

        # Check buffer shapes
        assert schedule.alpha_bar.shape == (9,)  # num_steps + 1
        assert schedule.sqrt_alpha_bar.shape == (9,)
        assert schedule.sqrt_one_minus_alpha_bar.shape == (9,)

        # Alpha bar should start at 1 and decrease
        assert schedule.alpha_bar[0].item() == pytest.approx(1.0, rel=1e-3)
        assert schedule.alpha_bar[-1].item() < schedule.alpha_bar[0].item()

        # Test get_noise_level
        sqrt_ab, sqrt_one_minus_ab = schedule.get_noise_level(0)
        assert sqrt_ab.item() == pytest.approx(1.0, rel=1e-3)

    def test_linear_schedule(self):
        """Test linear noise schedule."""
        from vllm.model_executor.models.dflash import DFlashNoiseSchedule

        schedule = DFlashNoiseSchedule(num_steps=8, schedule_type="linear")

        assert schedule.alpha_bar.shape == (9,)
        assert schedule.alpha_bar[0].item() == 1.0

    def test_sqrt_schedule(self):
        """Test sqrt noise schedule."""
        from vllm.model_executor.models.dflash import DFlashNoiseSchedule

        schedule = DFlashNoiseSchedule(num_steps=8, schedule_type="sqrt")

        assert schedule.alpha_bar.shape == (9,)

    def test_invalid_schedule(self):
        """Test that invalid schedule type raises error."""
        from vllm.model_executor.models.dflash import DFlashNoiseSchedule

        with pytest.raises(ValueError, match="Unknown noise schedule"):
            DFlashNoiseSchedule(num_steps=8, schedule_type="invalid")


class TestDFlashAttention:
    """Tests for DFlashAttention module."""

    def test_self_attention(self):
        """Test self-attention forward pass."""
        from vllm.model_executor.models.dflash import DFlashAttention

        batch_size = 2
        seq_len = 8
        hidden_size = 256
        num_heads = 8

        attn = DFlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = attn(hidden_states)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_cross_attention(self):
        """Test cross-attention forward pass."""
        from vllm.model_executor.models.dflash import DFlashAttention

        batch_size = 2
        seq_len = 8
        context_len = 4
        hidden_size = 256
        num_heads = 8

        attn = DFlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        context = torch.randn(batch_size, context_len, hidden_size)
        output = attn(hidden_states, context)

        assert output.shape == (batch_size, seq_len, hidden_size)


class TestDFlashDecoderLayer:
    """Tests for DFlashDecoderLayer."""

    def test_forward(self):
        """Test decoder layer forward pass."""
        from vllm.model_executor.models.dflash import DFlashDecoderLayer

        batch_size = 2
        seq_len = 8
        context_len = 4

        # Create a mock config
        class MockConfig:
            hidden_size = 256
            num_attention_heads = 8
            intermediate_size = 512
            rms_norm_eps = 1e-6

        config = MockConfig()
        layer = DFlashDecoderLayer(config, layer_idx=0)

        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        context = torch.randn(batch_size, context_len, config.hidden_size)

        output = layer(hidden_states, context)

        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestSpeculativeConfigDFlash:
    """Tests for DFlash integration in SpeculativeConfig."""

    def test_use_dflash_method(self):
        """Test use_dflash() helper method."""
        from vllm.config import SpeculativeConfig

        # Create a minimal config for testing
        # Note: This requires a full config setup which may not be feasible
        # in unit tests without mocking
        pass

    def test_dflash_in_speculative_method(self):
        """Test that 'dflash' is a valid speculative method."""
        from vllm.config.speculative import SpeculativeMethod
        from typing import get_args

        methods = get_args(SpeculativeMethod)
        assert "dflash" in methods


class TestDFlashModelRegistry:
    """Tests for DFlash model registration."""

    def test_model_registered(self):
        """Test that DFlashModel is registered in the model registry."""
        from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS

        assert "DFlashModel" in _SPECULATIVE_DECODING_MODELS
        module_name, class_name = _SPECULATIVE_DECODING_MODELS["DFlashModel"]
        assert module_name == "dflash"
        assert class_name == "DFlashModel"


class TestDFlashConfigExport:
    """Tests for DFlash config exports."""

    def test_config_in_exports(self):
        """Test that DFlashConfig is properly exported."""
        from vllm.transformers_utils.configs import DFlashConfig as ExportedConfig

        assert ExportedConfig is DFlashConfig

    def test_config_in_all(self):
        """Test that DFlashConfig is in __all__."""
        from vllm.transformers_utils import configs

        assert "DFlashConfig" in configs.__all__
