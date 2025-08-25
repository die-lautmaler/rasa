"""Utilities for automatic wandb sweep initialization and execution."""

import logging
import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class AutoSweepManager:
    """Manages automatic wandb sweep initialization and execution."""

    def __init__(self):
        """Initialize the auto sweep manager."""
        self.sweep_id = None
        self.is_sweep_run = False
        self.original_command = None

    def should_run_sweep(self, additional_arguments: Dict[str, Any]) -> bool:
        """Check if we should run an automatic sweep.
        
        Args:
            additional_arguments: Additional arguments from CLI.
            
        Returns:
            True if both sweep_script and sweep_config are provided.
        """
        sweep_script = additional_arguments.get("sweep_script")
        sweep_config = additional_arguments.get("sweep_config")
        
        return bool(sweep_script and sweep_config and WANDB_AVAILABLE)

    def initialize_and_run_sweep(
        self,
        sweep_config_path: str,
        sweep_script_path: str,
        training_args: Dict[str, Any]
    ) -> None:
        """Initialize a wandb sweep and run the agent automatically.
        
        Args:
            sweep_config_path: Path to the sweep configuration YAML file.
            sweep_script_path: Path to the sweep script Python file.
            training_args: Original training arguments.
        """
        if not WANDB_AVAILABLE:
            logger.error("wandb is not available. Cannot run sweep.")
            return

        try:
            # Load sweep configuration
            sweep_config = self._load_sweep_config(sweep_config_path)
            if not sweep_config:
                return

            # Create the sweep
            logger.info("Creating wandb sweep...")
            project_name = sweep_config.get('project', 'rasa-auto-sweep')
            self.sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
            
            logger.info(f"Created sweep: {self.sweep_id}")
            logger.info(f"View at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{self.sweep_id}")

            # Prepare the training function
            def train_with_sweep():
                self._run_training_iteration(sweep_script_path, training_args)

            # Run the sweep agent
            logger.info("Starting sweep agent...")
            count = sweep_config.get('run_count')  # Optional limit
            wandb.agent(self.sweep_id, function=train_with_sweep, count=count)
            
            logger.info("Sweep completed!")

        except Exception as e:
            logger.error(f"Failed to run auto sweep: {e}")
            sys.exit(1)

    def _load_sweep_config(self, sweep_config_path: str) -> Optional[Dict[str, Any]]:
        """Load sweep configuration from YAML file.
        
        Args:
            sweep_config_path: Path to the sweep configuration file.
            
        Returns:
            Sweep configuration dictionary or None if failed.
        """
        try:
            config_path = Path(sweep_config_path)
            if not config_path.exists():
                logger.error(f"Sweep config file not found: {sweep_config_path}")
                return None

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            logger.info(f"Loaded sweep config from {sweep_config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load sweep config: {e}")
            return None

    def _run_training_iteration(self, sweep_script_path: str, training_args: Dict[str, Any]) -> None:
        """Run a single training iteration for the sweep.
        
        Args:
            sweep_script_path: Path to the sweep script.
            training_args: Original training arguments.
        """
        logger.info("Starting sweep training iteration")
        
        # Initialize wandb run for this iteration
        wandb.init()
        
        try:
            # Call the training function directly with sweep parameters
            from rasa import model_training
            
            # Create a modified additional_arguments that includes sweep info
            modified_additional_args = training_args['additional_arguments'].copy()
            modified_additional_args['wandb'] = True  # Ensure wandb is enabled
            modified_additional_args['_is_sweep_iteration'] = True  # Mark as sweep iteration
            
            logger.info("Running training iteration with sweep parameters...")
            
            # Call train_nlu directly
            result = model_training.train_nlu(
                config=training_args['config'],
                nlu_data=training_args['nlu_data'],
                output=training_args['output'],
                fixed_model_name=training_args['fixed_model_name'],
                persist_nlu_training_data=training_args['persist_nlu_training_data'],
                additional_arguments=modified_additional_args,
                domain=training_args['domain'],
                model_to_finetune=training_args['model_to_finetune'],
                finetuning_epoch_fraction=training_args['finetuning_epoch_fraction'],
            )
            
            logger.info("Training iteration completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in sweep training iteration: {e}")
            # Log error to wandb
            if wandb.run:
                wandb.log({"training_error": 1, "error_message": str(e)})
            raise

        # Note: wandb.finish() will be called automatically by wandb.agent

    def is_auto_sweep_run(self) -> bool:
        """Check if this is an auto sweep run.
        
        Returns:
            True if this is an auto sweep run.
        """
        return self.is_sweep_run


# Global auto sweep manager instance
_auto_sweep_manager = AutoSweepManager()


def get_auto_sweep_manager() -> AutoSweepManager:
    """Get the global auto sweep manager instance.
    
    Returns:
        The global AutoSweepManager instance.
    """
    return _auto_sweep_manager


def should_initialize_auto_sweep(additional_arguments: Dict[str, Any]) -> bool:
    """Check if we should initialize an automatic sweep.
    
    Args:
        additional_arguments: Additional arguments from CLI.
        
    Returns:
        True if auto sweep should be initialized.
    """
    manager = get_auto_sweep_manager()
    return manager.should_run_sweep(additional_arguments)


def run_auto_sweep(
    sweep_config_path: str,
    sweep_script_path: str,
    training_args: Dict[str, Any]
) -> None:
    """Run automatic sweep initialization and execution.
    
    Args:
        sweep_config_path: Path to sweep configuration file.
        sweep_script_path: Path to sweep script file.
        training_args: Training arguments.
    """
    manager = get_auto_sweep_manager()
    manager.initialize_and_run_sweep(sweep_config_path, sweep_script_path, training_args)