from pathlib import Path
from typing import Dict, Text, Any, Optional

import logging
import tensorflow as tf
from tqdm import tqdm

import rasa.shared.utils.io

logger = logging.getLogger(__name__)


class RasaTrainingLogger(tf.keras.callbacks.Callback):
    """Callback for logging the status of training."""

    def __init__(self, epochs: int, silent: bool) -> None:
        """Initializes the callback.

        Args:
            epochs: Total number of epochs.
            silent: If 'True' the entire progressbar wrapper is disabled.
        """
        super().__init__()

        disable = silent or rasa.shared.utils.io.is_logging_disabled()
        self.progress_bar = tqdm(range(epochs), desc="Epochs", disable=disable)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Updates the logging output on every epoch end.

        Args:
            epoch: The current epoch.
            logs: The training metrics.
        """
        self.progress_bar.update(1)
        self.progress_bar.set_postfix(logs)

    def on_train_end(self, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Closes the progress bar after training.

        Args:
            logs: The training metrics.
        """
        self.progress_bar.close()


class RasaWandBLogger(tf.keras.callbacks.Callback):
    """Callback for logging training metrics to Weights & Biases."""

    def __init__(self, wandb_logger: Optional[Any] = None, log_frequency: int = 1000) -> None:
        """Initializes the callback.

        Args:
            wandb_logger: WandB logger instance for logging metrics.
                         If None, will try to get from thread-local context.
            log_frequency: Frequency for logging training metrics (in steps).
        """
        super().__init__()
        self.wandb_logger = wandb_logger
        self.current_epoch = 0
        self.batches_per_epoch = 0
        self.global_step = 0  # Global step counter for monotonic logging
        self.log_frequency = log_frequency  # Log training metrics every N steps
        
        # Initialize previous validation metrics for comparison
        if self.wandb_logger:
            self.wandb_logger._previous_val_metrics = {}

    def on_batch_end(self, batch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Logs training metrics to wandb at specified frequency.

        Args:
            batch: The current batch.
            logs: The training metrics.
        """
        from rasa.utils.wandb_utils import get_current_wandb_logger
        
        # Use provided wandb_logger or get from thread-local context
        wandb_logger = self.wandb_logger or get_current_wandb_logger()
        
        if wandb_logger is None or logs is None:
            return

        # Increment global step counter
        self.global_step += 1

        # Only log training metrics every LOG_FREQUENCY steps
        if self.global_step % self.log_frequency != 0:
            return

        # Log training metrics only (exclude validation metrics)
        try:
            # Map common metric names to more readable ones
            metric_mapping = {
                # Loss metrics
                't_loss': 'train/total_loss',
                'i_loss': 'train/intent_loss',
                'e_loss': 'train/entity_loss',
                'g_loss': 'train/entity_group_loss',
                'r_loss': 'train/entity_role_loss',
                'm_loss': 'train/mask_loss',
                
                # Accuracy metrics
                'i_acc': 'train/intent_accuracy', 
                'm_acc': 'train/mask_accuracy',
                
                # F1 metrics
                'e_f1': 'train/entity_f1_score',
                'g_f1': 'train/entity_group_f1_score',
                'r_f1': 'train/entity_role_f1_score',
                
                # Legacy mappings for backward compatibility
                'loss': 'train/loss'
            }
            
            # Prepare training metrics for logging (exclude validation metrics)
            wandb_metrics = {}
            
            for key, value in logs.items():
                # Skip validation metrics - these will be logged in on_epoch_end
                if key.startswith('val_'):
                    continue
                    
                # Only log numeric values
                if not isinstance(value, (int, float)) and not hasattr(value, 'item'):
                    continue
                    
                # Convert to float if it's a numpy scalar
                numeric_value = value.item() if hasattr(value, 'item') else float(value)
                
                # Use mapped name if available, otherwise use original key
                metric_name = metric_mapping.get(key, key)
                wandb_metrics[metric_name] = numeric_value
            
            # Only log if we have training metrics to log
            if wandb_metrics:
                wandb_logger.log_metrics(wandb_metrics, step=self.global_step)
            
        except Exception as e:
            logger.debug(f"Failed to log training metrics to wandb: {e}")

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Track epoch start for step calculation.

        Args:
            epoch: The current epoch.
            logs: The training metrics.
        """
        self.current_epoch = epoch
        # Get batches per epoch from the training params if available
        if hasattr(self.params, 'steps'):
            self.batches_per_epoch = self.params['steps']

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Logs validation metrics to wandb after epoch completion.

        Args:
            epoch: The current epoch.
            logs: The training metrics.
        """
        from rasa.utils.wandb_utils import get_current_wandb_logger
        
        # Use provided wandb_logger or get from thread-local context
        wandb_logger = self.wandb_logger or get_current_wandb_logger()
        
        if wandb_logger is None or logs is None:
            return

        # Log only validation metrics at epoch end
        try:
            # Map validation metric names to more readable ones
            metric_mapping = {
                # Validation metrics
                'val_t_loss': 'validation/total_loss',
                'val_i_loss': 'validation/intent_loss',
                'val_e_loss': 'validation/entity_loss',
                'val_g_loss': 'validation/entity_group_loss',
                'val_r_loss': 'validation/entity_role_loss',
                'val_m_loss': 'validation/mask_loss',
                'val_i_acc': 'validation/intent_accuracy',
                'val_m_acc': 'validation/mask_accuracy',
                'val_e_f1': 'validation/entity_f1_score',
                'val_g_f1': 'validation/entity_group_f1_score',
                'val_r_f1': 'validation/entity_role_f1_score',
                
                # Legacy mappings for backward compatibility
                'val_loss': 'validation/loss'
            }
            
            # Prepare validation metrics for logging
            wandb_metrics = {}
            validation_metrics = {}
            
            for key, value in logs.items():
                # Only process validation metrics
                if not key.startswith('val_'):
                    continue
                    
                # Only log numeric values
                if not isinstance(value, (int, float)) and not hasattr(value, 'item'):
                    continue
                    
                # Convert to float if it's a numpy scalar
                numeric_value = value.item() if hasattr(value, 'item') else float(value)
                
                # Use mapped name if available, otherwise use original key
                metric_name = metric_mapping.get(key, key)
                wandb_metrics[metric_name] = numeric_value
                
                # Store for comparison tracking
                validation_metrics[key.replace('val_', '')] = numeric_value
            
            # Only log if we have validation metrics
            if wandb_metrics:
                # Add epoch number for validation metrics
                wandb_metrics['validation/epoch'] = epoch + 1  # Make it 1-based
                
                # Add improvement tracking if we have previous validation metrics
                if hasattr(wandb_logger, '_previous_val_metrics') and wandb_logger._previous_val_metrics:
                    improving_count = 0
                    total_val_metrics = 0
                    for metric, current_val in validation_metrics.items():
                        if metric in wandb_logger._previous_val_metrics:
                            total_val_metrics += 1
                            # For loss metrics, improvement means decrease
                            # For accuracy/f1 metrics, improvement means increase
                            if 'loss' in metric:
                                if current_val < wandb_logger._previous_val_metrics[metric]:
                                    improving_count += 1
                            else:  # accuracy or f1 metrics
                                if current_val > wandb_logger._previous_val_metrics[metric]:
                                    improving_count += 1
                    
                    if total_val_metrics > 0:
                        wandb_metrics['validation/metrics_improving_ratio'] = improving_count / total_val_metrics
                
                # Store current validation metrics for next comparison
                wandb_logger._previous_val_metrics = validation_metrics.copy()
                
                # Log validation metrics without step to avoid conflicts with training steps
                # WandB will automatically handle the timeline, and we can use epoch info in metric names
                # Add epoch information to the metrics for better tracking
                epoch_metrics = wandb_metrics.copy()
                epoch_metrics['validation/epoch'] = epoch + 1
                
                wandb_logger.log_metrics(epoch_metrics, step=None)
            
        except Exception as e:
            logger.debug(f"Failed to log validation metrics to wandb: {e}")


class RasaModelCheckpoint(tf.keras.callbacks.Callback):
    """Callback for saving intermediate model checkpoints."""

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initializes the callback.

        Args:
            checkpoint_dir: Directory to store checkpoints to.
        """
        super().__init__()

        self.checkpoint_file = checkpoint_dir / "checkpoint.tf_model"
        self.best_metrics_so_far: Dict[Text, Any] = {}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Save the model on epoch end if the model has improved.

        Args:
            epoch: The current epoch.
            logs: The training metrics.
        """
        if self._does_model_improve(logs):
            logger.debug(f"Creating model checkpoint at epoch={epoch + 1} ...")
            self.model.save_weights(
                self.checkpoint_file, overwrite=True, save_format="tf"
            )

    def _does_model_improve(self, curr_results: Dict[Text, Any]) -> bool:
        """Checks whether the current results are better than the best so far.

        Results are considered better if each metric is equal or better than the best so
        far, and at least one is better.

        Args:
            curr_results: The training metrics for this epoch.
        """
        curr_metric_names = [
            k
            for k in curr_results.keys()
            if k.startswith("val") and (k.endswith("_acc") or k.endswith("_f1"))
        ]
        # the "val" prefix is prepended to metrics in fit if _should_eval returns true
        # for this particular epoch
        if len(curr_metric_names) == 0:
            # the metrics are not validation metrics
            return False
        # initialize best_metrics_so_far with the first results
        if not self.best_metrics_so_far:
            for metric_name in curr_metric_names:
                self.best_metrics_so_far[metric_name] = float(curr_results[metric_name])
            return True

        at_least_one_improved = False
        improved_metrics = {}
        for metric_name in self.best_metrics_so_far.keys():
            if float(curr_results[metric_name]) < self.best_metrics_so_far[metric_name]:
                # at least one of the values is worse
                return False
            if float(curr_results[metric_name]) > self.best_metrics_so_far[metric_name]:
                at_least_one_improved = True
                improved_metrics[metric_name] = float(curr_results[metric_name])

        # all current values >= previous best and at least one is better
        if at_least_one_improved:
            self.best_metrics_so_far.update(improved_metrics)
        return at_least_one_improved
