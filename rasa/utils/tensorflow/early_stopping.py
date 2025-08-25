"""Early stopping callback for Rasa training."""

import logging
from typing import Dict, Text, Any, Optional, List, Union

import tensorflow as tf

logger = logging.getLogger(__name__)


class RasaEarlyStopping(tf.keras.callbacks.Callback):
    """Early stopping callback for Rasa model training.
    
    Monitors a specified metric and stops training when the metric has stopped
    improving by a minimum delta for a specified number of epochs.
    
    This callback is similar to tf.keras.callbacks.EarlyStopping but designed
    specifically for Rasa's metric naming conventions and logging patterns.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 10,
        verbose: bool = True,
        mode: str = "auto",
        baseline: Optional[float] = None,
        restore_best_weights: bool = False,
    ) -> None:
        """Initializes the early stopping callback.

        Args:
            monitor: Metric to be monitored. Can be any metric logged during training
                    (e.g., 'val_loss', 'val_i_acc', 'val_e_f1', 'loss', 'i_acc').
            min_delta: Minimum change in the monitored quantity to qualify as an
                      improvement. For example, an absolute change of less than 
                      min_delta will count as no improvement.
            patience: Number of epochs with no improvement after which training
                     will be stopped.
            verbose: Whether to print messages when early stopping is triggered.
            mode: One of {'auto', 'min', 'max'}. In 'min' mode, training will
                 stop when the monitored quantity has stopped decreasing; in 'max'
                 mode it will stop when the monitored quantity has stopped increasing;
                 in 'auto' mode, the direction is automatically inferred from the
                 name of the monitored quantity.
            baseline: Baseline value for the monitored quantity. Training will
                     stop if the model doesn't show improvement over the baseline.
            restore_best_weights: Whether to restore model weights from the epoch
                                with the best value of the monitored quantity.
        """
        super().__init__()
        
        self.monitor = monitor
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        
        # Automatically infer the mode if set to 'auto'
        if mode not in ['auto', 'min', 'max']:
            logger.warning(f"EarlyStopping mode {mode} is unknown, fallback to auto mode.")
            mode = 'auto'
            
        if mode == 'auto':
            # Infer mode from metric name
            if 'loss' in monitor.lower():
                self.mode = 'min'
            elif any(metric in monitor.lower() for metric in ['acc', 'f1', 'precision', 'recall']):
                self.mode = 'max'
            else:
                logger.warning(
                    f"Could not infer mode for metric '{monitor}'. "
                    "Defaulting to 'min' mode. Consider setting mode explicitly."
                )
                self.mode = 'min'
        else:
            self.mode = mode
            
        # Set comparison operators based on mode
        if self.mode == 'min':
            self.monitor_op = lambda current, best: (current - self.min_delta) < best
            self.min_delta *= -1
        else:  # mode == 'max'
            self.monitor_op = lambda current, best: (current - self.min_delta) > best
            
        # Initialize tracking variables
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        self.best_epoch = 0

    def on_train_begin(self, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Called at the beginning of training."""
        self.wait = 0
        self.stopped_epoch = 0
        
        # Initialize best value based on mode
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = float('inf') if self.mode == 'min' else float('-inf')
            
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Called at the end of each epoch."""
        if logs is None:
            return
            
        # Get the monitored metric value
        current = self.get_monitor_value(logs)
        if current is None:
            return
            
        # Check if this is the best value so far
        if self.monitor_op(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            
            # Store the best weights if requested
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
                
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.monitor} improved from {self.best:.6f} "
                    f"to {current:.6f} at epoch {epoch + 1}"
                )
        else:
            self.wait += 1
            if self.verbose and self.wait > 0:
                logger.info(
                    f"EarlyStopping: {self.monitor} did not improve from {self.best:.6f} "
                    f"(current: {current:.6f}, patience: {self.wait}/{self.patience})"
                )
                
        # Check if we should stop training
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose:
                    logger.info(
                        f"EarlyStopping: Restoring model weights from epoch {self.best_epoch + 1}"
                    )
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Called at the end of training."""
        if self.stopped_epoch > 0:
            if self.verbose:
                logger.info(
                    f"EarlyStopping: Training stopped at epoch {self.stopped_epoch + 1}. "
                    f"Best {self.monitor} was {self.best:.6f} at epoch {self.best_epoch + 1}."
                )
        elif self.verbose:
            logger.info(
                f"EarlyStopping: Training completed. "
                f"Best {self.monitor} was {self.best:.6f} at epoch {self.best_epoch + 1}."
            )

    def get_monitor_value(self, logs: Dict[Text, Any]) -> Optional[float]:
        """Extract the monitored metric value from logs.
        
        Args:
            logs: Dictionary containing metric values.
            
        Returns:
            The value of the monitored metric, or None if not found.
        """
        monitor_value = logs.get(self.monitor)
        
        if monitor_value is None:
            # Log available metrics for debugging
            available_metrics = list(logs.keys())
            logger.warning(
                f"EarlyStopping: Metric '{self.monitor}' not found in logs. "
                f"Available metrics: {available_metrics}"
            )
            return None
            
        # Handle different value types (numpy scalars, tensors, etc.)
        if hasattr(monitor_value, 'item'):
            return float(monitor_value.item())
        elif isinstance(monitor_value, (int, float)):
            return float(monitor_value)
        else:
            logger.warning(
                f"EarlyStopping: Could not convert {self.monitor} value "
                f"{monitor_value} (type: {type(monitor_value)}) to float"
            )
            return None

    def get_config(self) -> Dict[Text, Any]:
        """Returns the configuration of the callback as a dictionary."""
        return {
            'monitor': self.monitor,
            'min_delta': self.min_delta,
            'patience': self.patience,
            'verbose': self.verbose,
            'mode': self.mode,
            'baseline': self.baseline,
            'restore_best_weights': self.restore_best_weights,
        }


def create_early_stopping_callback(
    early_stopping_config: Dict[Text, Any]
) -> Optional[RasaEarlyStopping]:
    """Create an early stopping callback from configuration.
    
    Args:
        early_stopping_config: Configuration dictionary containing early stopping parameters.
        
    Returns:
        RasaEarlyStopping callback instance, or None if early stopping is disabled.
        
    Example config:
        early_stopping:
          enabled: true
          monitor: "val_loss"
          min_delta: 0.001
          patience: 10
          mode: "auto"
          restore_best_weights: false
    """
    if not early_stopping_config or not early_stopping_config.get('enabled', False):
        return None
        
    # Extract parameters with defaults
    monitor = early_stopping_config.get('monitor', 'val_loss')
    min_delta = early_stopping_config.get('min_delta', 0.0)
    patience = early_stopping_config.get('patience', 10)
    verbose = early_stopping_config.get('verbose', True)
    mode = early_stopping_config.get('mode', 'auto')
    baseline = early_stopping_config.get('baseline')
    restore_best_weights = early_stopping_config.get('restore_best_weights', False)
    
    # Validate parameters
    if patience <= 0:
        logger.warning(f"Early stopping patience must be > 0, got {patience}. Disabling early stopping.")
        return None
        
    if min_delta < 0:
        logger.warning(f"Early stopping min_delta must be >= 0, got {min_delta}. Using 0.")
        min_delta = 0.0
        
    logger.info(
        f"Early stopping enabled: monitor={monitor}, patience={patience}, "
        f"min_delta={min_delta}, mode={mode}, restore_best_weights={restore_best_weights}"
    )
    
    return RasaEarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        verbose=verbose,
        mode=mode,
        baseline=baseline,
        restore_best_weights=restore_best_weights,
    )