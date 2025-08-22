"""Utilities for Weights & Biases integration."""

import logging
from typing import Any, Dict, Optional, Text, List
import functools
import os

logger = logging.getLogger(__name__)

# Flag to track if wandb is available
WANDB_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    logger.debug(
        "Weights & Biases (wandb) is not installed. To use wandb logging, "
        "install rasa with 'pip install rasa[wandb]'"
    )
    wandb = None


def _check_wandb_availability() -> bool:
    """Check if wandb is available and properly configured.

    Returns:
        True if wandb is available and can be used, False otherwise.
    """
    if not WANDB_AVAILABLE or wandb is None:
        logger.warning(
            "Weights & Biases is not available. Install it with 'pip install rasa[wandb]' "
            "to enable wandb logging."
        )
        return False

    # Check if wandb is logged in
    if not wandb.api.api_key:
        logger.warning(
            "W&B API key not found. Please run 'wandb login' or set the WANDB_API_KEY "
            "environment variable to enable wandb logging."
        )
        return False

    return True


def requires_wandb(func):
    """Decorator to check if wandb is available before calling a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _check_wandb_availability():
            return None
        return func(*args, **kwargs)

    return wrapper


class WandBLogger:
    """Weights & Biases logger for Rasa NLU training."""

    def __init__(self, project_name: str = "rasa-nlu", **kwargs):
        """Initialize wandb logger.

        Args:
            project_name: Name of the wandb project.
            **kwargs: Additional arguments to pass to wandb.init().
        """
        self.project_name = project_name
        self.run = None
        self.config = kwargs

    @requires_wandb
    def init_run(
        self, config: Optional[Dict[Text, Any]] = None, run_name: Optional[str] = None
    ) -> None:
        """Initialize a wandb run.

        Args:
            config: Configuration to log to wandb.
            run_name: Name for the wandb run.
        """
        try:
            wandb_config = {
                "project": self.project_name,
                "name": run_name,
                "job_type": "train",
                **self.config,
            }

            if config:
                wandb_config["config"] = config

            self.run = wandb.init(**wandb_config)
            logger.info(f"Started wandb run: {self.run.url}")

        except Exception as e:
            logger.warning(f"Failed to initialize wandb run: {e}")
            self.run = None

    @requires_wandb
    def log_metrics(self, metrics: Dict[Text, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb.

        Args:
            metrics: Dictionary of metrics to log.
            step: Training step/epoch number.
        """
        if self.run is None:
            return

        try:
            wandb_metrics = {}
            for key, value in metrics.items():
                # Handle different types of values
                if isinstance(value, (int, float)):
                    wandb_metrics[key] = value
                elif hasattr(value, "item"):  # Handle numpy scalars
                    wandb_metrics[key] = value.item()
                else:
                    # Convert other types to string
                    wandb_metrics[f"{key}_str"] = str(value)

            if step is not None:
                wandb_metrics["step"] = step

            wandb.log(wandb_metrics, step=step)

        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {e}")

    @requires_wandb
    def log_config(self, config: Dict[Text, Any]) -> None:
        """Log configuration to wandb.

        Args:
            config: Configuration dictionary to log.
        """
        if self.run is None:
            return

        try:
            wandb.config.update(config)
        except Exception as e:
            logger.warning(f"Failed to log config to wandb: {e}")

    @requires_wandb
    def log_artifact(
        self, artifact_path: Text, artifact_name: str, artifact_type: str = "model"
    ) -> None:
        """Log an artifact to wandb.

        Args:
            artifact_path: Path to the artifact.
            artifact_name: Name for the artifact.
            artifact_type: Type of artifact (e.g., 'model', 'dataset').
        """
        if self.run is None:
            return

        try:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            if os.path.isfile(artifact_path):
                artifact.add_file(artifact_path)
            elif os.path.isdir(artifact_path):
                artifact.add_dir(artifact_path)
            else:
                logger.warning(f"Artifact path does not exist: {artifact_path}")
                return

            wandb.log_artifact(artifact)
            logger.info(f"Logged artifact '{artifact_name}' to wandb")

        except Exception as e:
            logger.warning(f"Failed to log artifact to wandb: {e}")

    @requires_wandb
    def finish_run(self) -> None:
        """Finish the wandb run."""
        if self.run is None:
            return

        try:
            wandb.finish()
            logger.info("Finished wandb run")
        except Exception as e:
            logger.warning(f"Failed to finish wandb run: {e}")
        finally:
            self.run = None


def create_wandb_logger(
    config: Optional[Dict[Text, Any]] = None, run_name: Optional[str] = None
) -> Optional[WandBLogger]:
    """Create and initialize a WandB logger.

    Args:
        config: Configuration to log to wandb.
        run_name: Name for the wandb run.

    Returns:
        WandBLogger instance if wandb is available, None otherwise.
    """
    if not _check_wandb_availability():
        return None

    logger_instance = WandBLogger()
    logger_instance.init_run(config=config, run_name=run_name)
    return logger_instance


def extract_training_metrics(
    training_data: Any, model_configuration: Any
) -> Dict[Text, Any]:
    """Extract metrics from training data and model configuration.

    Args:
        training_data: The NLU training data.
        model_configuration: The model configuration.

    Returns:
        Dictionary of metrics to log.
    """
    metrics = {}

    try:
        # Extract training data metrics
        if hasattr(training_data, "training_examples"):
            metrics["num_training_examples"] = len(training_data.training_examples)

        if hasattr(training_data, "intents"):
            metrics["num_intents"] = len(training_data.intents)

        if hasattr(training_data, "entities"):
            metrics["num_entities"] = len(training_data.entities)

        if hasattr(training_data, "entity_synonyms"):
            metrics["num_entity_synonyms"] = len(training_data.entity_synonyms)

        if hasattr(training_data, "regex_features"):
            metrics["num_regex_features"] = len(training_data.regex_features)

        if hasattr(training_data, "lookup_tables"):
            metrics["num_lookup_tables"] = len(training_data.lookup_tables)

        # Extract model configuration metrics
        if model_configuration and hasattr(model_configuration, "as_dict"):
            config_dict = model_configuration.as_dict()
            if "pipeline" in config_dict:
                metrics["num_pipeline_components"] = len(config_dict["pipeline"])

            # Extract component-specific configs
            for component in config_dict.get("pipeline", []):
                component_name = component.get("name", "").replace(".", "_")
                for key, value in component.items():
                    if key != "name" and isinstance(value, (int, float, bool)):
                        metrics[f"{component_name}_{key}"] = value

    except Exception as e:
        logger.debug(f"Failed to extract some training metrics: {e}")

    return metrics
