"""Tests for model module."""

import torch
import pytest

from src.model import CNNLSTMMultiTask


@pytest.fixture
def model():
    return CNNLSTMMultiTask(
        num_channels=7,
        window_size=128,
        num_classes=8,
        conv_channels=[32, 64],
        conv_kernel=3,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        lstm_bidirectional=True,
        fc_hidden_size=64,
        dropout=0.3,
    )


class TestCNNLSTMMultiTask:
    def test_output_shapes(self, model):
        x = torch.randn(4, 7, 128)
        cls_logits, knee_pred = model(x)

        assert cls_logits.shape == (4, 8)
        assert knee_pred.shape == (4, 1)

    def test_forward_pass_runs(self, model):
        """Smoke test: forward pass should not crash."""
        x = torch.randn(16, 7, 128)
        cls_logits, knee_pred = model(x)
        # Both outputs should be finite
        assert torch.isfinite(cls_logits).all()
        assert torch.isfinite(knee_pred).all()

    def test_gradients_flow(self, model):
        """Backward should compute gradients for all parameters."""
        x = torch.randn(8, 7, 128)
        cls_logits, knee_pred = model(x)

        loss = cls_logits.mean() + knee_pred.mean()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient for {name}"

    def test_different_input_sizes(self, model):
        """Model should accept variable batch sizes."""
        for bs in [1, 8, 32]:
            x = torch.randn(bs, 7, 128)
            cls_logits, knee_pred = model(x)
            assert cls_logits.shape[0] == bs
            assert knee_pred.shape[0] == bs

    def test_batch_size_one(self, model):
        x = torch.randn(1, 7, 128)
        cls_logits, knee_pred = model(x)
        assert cls_logits.shape == (1, 8)
        assert knee_pred.shape == (1, 1)

    def test_eval_mode_no_dropout_variation(self, model):
        """In eval mode, same input should give same output."""
        model.eval()
        x = torch.randn(4, 7, 128)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1[0], out2[0])
        assert torch.equal(out1[1], out2[1])
