"""Component tests for training callback."""

from __future__ import annotations

import os

from unittest.mock import MagicMock, patch

from deepfabric.training.api_key_prompt import (
    clear_api_key_cache,
)
from deepfabric.training.callback import DeepFabricCallback
from tests.support import training_callback as _support

mock_metrics_http = _support.mock_metrics_http


class TestDeepFabricCallback:
    """Tests for DeepFabricCallback class."""

    def test_callback_enabled_with_api_key(self):
        """Callback should be enabled when API key provided."""
        callback = DeepFabricCallback(api_key="test-key")

        assert callback.enabled
        assert callback.sender.enabled
        assert callback.run_id is not None

        callback.sender.shutdown()

    def test_callback_uses_env_api_key(self):
        """Callback should use DEEPFABRIC_API_KEY env var."""
        os.environ["DEEPFABRIC_API_KEY"] = "env-test-key"
        clear_api_key_cache()

        try:
            callback = DeepFabricCallback()

            assert callback.enabled
            assert callback.api_key == "env-test-key"

            callback.sender.shutdown()
        finally:
            del os.environ["DEEPFABRIC_API_KEY"]
            clear_api_key_cache()

    def test_callback_uses_custom_endpoint(self):
        """Callback should use custom endpoint if provided."""
        callback = DeepFabricCallback(
            api_key="test-key",
            endpoint="https://custom.api.com",
        )

        assert callback.endpoint == "https://custom.api.com"

        callback.sender.shutdown()

    def test_callback_accepts_trainer(self):
        """Callback should accept trainer instance."""
        mock_trainer = MagicMock()
        mock_trainer.model = MagicMock()
        mock_trainer.model.config.name_or_path = "test-model"

        callback = DeepFabricCallback(trainer=mock_trainer, api_key="test-key")

        assert callback._trainer is mock_trainer

        callback.sender.shutdown()

    def test_on_log_filters_none_values(self):
        """on_log should filter out None values from logs."""
        callback = DeepFabricCallback(api_key="test-key")

        # Mock state and args
        mock_state = MagicMock()
        mock_state.global_step = 100
        mock_state.epoch = 0.5
        mock_args = MagicMock()
        mock_control = MagicMock()

        logs = {"loss": 2.5, "learning_rate": 0.001, "grad_norm": None}

        callback.on_log(mock_args, mock_state, mock_control, logs=logs)

        # Check that metrics were queued
        assert callback.sender._queue.qsize() == 1

        callback.sender.shutdown()

    def test_on_log_skips_empty_logs(self):
        """on_log should skip if logs is empty or None."""
        callback = DeepFabricCallback(api_key="test-key")

        mock_state = MagicMock()
        mock_state.global_step = 100
        mock_args = MagicMock()
        mock_control = MagicMock()

        # Test with None
        callback.on_log(mock_args, mock_state, mock_control, logs=None)
        assert callback.sender._queue.qsize() == 0

        callback.sender.shutdown()

    def test_on_train_begin_sends_run_start(self):
        """on_train_begin should send run start event."""
        callback = DeepFabricCallback(api_key="test-key")

        mock_state = MagicMock()
        mock_state.max_steps = 1000
        mock_state.num_train_epochs = 3
        mock_state.is_world_process_zero = True

        mock_args = MagicMock()
        mock_args.num_train_epochs = 3
        mock_args.per_device_train_batch_size = 8
        mock_args.learning_rate = 2e-5

        mock_control = MagicMock()

        callback.on_train_begin(mock_args, mock_state, mock_control)

        # Check that run_start was queued
        assert callback._run_started
        assert callback.sender._queue.qsize() == 1

        callback.sender.shutdown()

    def test_on_train_end_sends_run_end(self):
        """on_train_end should send run end event."""
        callback = DeepFabricCallback(api_key="test-key")
        callback._run_started = True

        mock_state = MagicMock()
        mock_state.global_step = 1000
        mock_state.epoch = 3.0
        mock_state.total_flos = 1e15
        mock_state.best_metric = 0.95

        mock_args = MagicMock()
        mock_control = MagicMock()

        with patch.object(callback.sender, "flush"):
            callback.on_train_end(mock_args, mock_state, mock_control)

        # Check that run_end was queued
        assert callback.sender._queue.qsize() == 1

        callback.sender.shutdown()

    def test_callback_has_all_required_methods(self):
        """Callback should have all TrainerCallback methods."""
        callback = DeepFabricCallback(api_key="test-key")

        # Required methods
        required_methods = [
            "on_train_begin",
            "on_train_end",
            "on_log",
            "on_evaluate",
            "on_save",
            "on_step_begin",
            "on_step_end",
            "on_epoch_begin",
            "on_epoch_end",
            "on_pre_optimizer_step",
        ]

        for method in required_methods:
            assert hasattr(callback, method), f"Missing method: {method}"
            assert callable(getattr(callback, method))

        callback.sender.shutdown()


class TestIntegrationWithMockTrainer:
    """Component tests with a mock Trainer and real metrics sender."""

    def test_callback_integration_with_mock_trainer(self):
        """Test callback works with a mock Trainer-like object."""
        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer.model = MagicMock()
        mock_trainer.model.config.name_or_path = "test-model"

        # Create callback with trainer
        callback = DeepFabricCallback(trainer=mock_trainer, api_key="test-key")

        # Mock trainer components
        mock_args = MagicMock()
        mock_args.num_train_epochs = 3
        mock_args.per_device_train_batch_size = 8
        mock_args.learning_rate = 2e-5
        mock_args.warmup_steps = 100
        mock_args.output_dir = "./output"

        mock_state = MagicMock()
        mock_state.max_steps = 1000
        mock_state.num_train_epochs = 3
        mock_state.global_step = 0
        mock_state.epoch = 0.0
        mock_state.is_world_process_zero = True

        mock_control = MagicMock()

        # Simulate training lifecycle
        callback.on_train_begin(mock_args, mock_state, mock_control)
        assert callback._run_started

        # Simulate logging
        for step in range(10):
            mock_state.global_step = step
            mock_state.epoch = step / 100
            logs = {"loss": 2.5 - step * 0.1, "learning_rate": 2e-5}
            callback.on_log(mock_args, mock_state, mock_control, logs=logs)

        # Simulate evaluation
        eval_metrics = {"eval_loss": 2.0, "eval_accuracy": 0.85}
        callback.on_evaluate(mock_args, mock_state, mock_control, metrics=eval_metrics)

        # Simulate training end
        mock_state.global_step = 1000
        mock_state.epoch = 3.0

        with patch.object(callback.sender, "flush"):
            callback.on_train_end(mock_args, mock_state, mock_control)

        callback.sender.shutdown()
