"""Utilities for loading and executing sweep scripts for Rasa training."""

import logging
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Text, Callable

logger = logging.getLogger(__name__)


class SweepScriptLoader:
    """Loads and executes sweep scripts for parameter injection."""

    def __init__(self):
        """Initialize the sweep script loader."""
        self.script_path = None
        self.script_module = None
        self.config_modifier_func = None

    def load_sweep_script(self, script_path: Text) -> bool:
        """Load a sweep script from the given path.
        
        Args:
            script_path: Path to the Python sweep script file.
            
        Returns:
            True if the script was loaded successfully, False otherwise.
        """
        try:
            script_path = Path(script_path)
            
            if not script_path.exists():
                logger.error(f"Sweep script not found: {script_path}")
                return False
                
            if not script_path.suffix == '.py':
                logger.error(f"Sweep script must be a Python file: {script_path}")
                return False
                
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("sweep_script", script_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load sweep script: {script_path}")
                return False
                
            self.script_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.script_module)
            
            # Look for the required function
            if hasattr(self.script_module, 'modify_config_for_sweep'):
                self.config_modifier_func = self.script_module.modify_config_for_sweep
                logger.info(f"Successfully loaded sweep script: {script_path}")
                self.script_path = str(script_path)
                return True
            else:
                logger.error(
                    f"Sweep script must contain a 'modify_config_for_sweep' function: {script_path}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Failed to load sweep script {script_path}: {e}")
            return False

    def modify_config(self, config: Dict[Text, Any]) -> Dict[Text, Any]:
        """Apply sweep modifications to the config.
        
        Args:
            config: Original Rasa configuration dictionary.
            
        Returns:
            Modified configuration dictionary.
        """
        if self.config_modifier_func is None:
            logger.warning("No sweep script loaded, returning original config")
            return config
            
        try:
            logger.info("Applying sweep script modifications to config")
            modified_config = self.config_modifier_func(config)
            
            if not isinstance(modified_config, dict):
                logger.error("Sweep script must return a dictionary")
                return config
                
            logger.info("Successfully applied sweep script modifications")
            return modified_config
            
        except Exception as e:
            logger.error(f"Failed to apply sweep script modifications: {e}")
            return config

    def is_loaded(self) -> bool:
        """Check if a sweep script is loaded.
        
        Returns:
            True if a sweep script is loaded, False otherwise.
        """
        return self.config_modifier_func is not None

    def get_script_path(self) -> Optional[str]:
        """Get the path of the loaded script.
        
        Returns:
            Path to the loaded script or None if no script is loaded.
        """
        return self.script_path


# Global sweep script loader instance
_sweep_script_loader = SweepScriptLoader()


def get_sweep_script_loader() -> SweepScriptLoader:
    """Get the global sweep script loader instance.
    
    Returns:
        The global SweepScriptLoader instance.
    """
    return _sweep_script_loader


def create_sweep_script_template() -> str:
    """Create a template sweep script content.
    
    Returns:
        Template sweep script as a string.
    """
    return '''"""
Example Rasa Sweep Script Template

This script demonstrates how to create a sweep script for Rasa training.
The script must contain a 'modify_config_for_sweep' function that takes
a config dictionary and returns a modified config dictionary.

You can use wandb.config to access sweep parameters, or implement your
own parameter sampling logic.
"""

import logging
from typing import Dict, Text, Any

logger = logging.getLogger(__name__)

# Import wandb if available for sweep integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def modify_config_for_sweep(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """Modify the Rasa config for sweep experiments.
    
    This function is called before training starts and allows you to:
    1. Access wandb sweep parameters via wandb.config
    2. Modify any part of the Rasa configuration
    3. Implement custom parameter sampling logic
    
    Args:
        config: Original Rasa configuration dictionary
        
    Returns:
        Modified configuration dictionary
    """
    # Create a copy to avoid modifying the original
    import copy
    modified_config = copy.deepcopy(config)
    
    logger.info("Applying sweep modifications to Rasa config")
    
    # Example 1: Using wandb sweep parameters
    if WANDB_AVAILABLE and wandb.config:
        logger.info("Detected wandb sweep parameters")
        
        # DIET Classifier parameters
        if hasattr(wandb.config, 'diet_epochs'):
            set_component_param(modified_config, 'DIETClassifier', 'epochs', wandb.config.diet_epochs)
            
        if hasattr(wandb.config, 'diet_learning_rate'):
            set_component_param(modified_config, 'DIETClassifier', 'learning_rate', wandb.config.diet_learning_rate)
            
        if hasattr(wandb.config, 'diet_batch_size'):
            set_component_param(modified_config, 'DIETClassifier', 'batch_size', wandb.config.diet_batch_size)
            
        if hasattr(wandb.config, 'diet_transformer_size'):
            set_component_param(modified_config, 'DIETClassifier', 'transformer_size', wandb.config.diet_transformer_size)
            
        if hasattr(wandb.config, 'diet_num_transformer_layers'):
            set_component_param(modified_config, 'DIETClassifier', 'number_of_transformer_layers', wandb.config.diet_num_transformer_layers)
            
        if hasattr(wandb.config, 'diet_drop_rate'):
            set_component_param(modified_config, 'DIETClassifier', 'drop_rate', wandb.config.diet_drop_rate)
            
        # Early stopping parameters
        if hasattr(wandb.config, 'early_stopping_patience'):
            set_component_param(modified_config, 'DIETClassifier', 'early_stopping.patience', wandb.config.early_stopping_patience)
            
        if hasattr(wandb.config, 'early_stopping_monitor'):
            set_component_param(modified_config, 'DIETClassifier', 'early_stopping.monitor', wandb.config.early_stopping_monitor)
            
        # TED Policy parameters
        if hasattr(wandb.config, 'ted_epochs'):
            set_policy_param(modified_config, 'UnexpecTEDIntentPolicy', 'epochs', wandb.config.ted_epochs)
            
        if hasattr(wandb.config, 'ted_learning_rate'):
            set_policy_param(modified_config, 'UnexpecTEDIntentPolicy', 'learning_rate', wandb.config.ted_learning_rate)
    
    # Example 2: Custom parameter sampling (if not using wandb sweeps)
    else:
        logger.info("Using custom parameter sampling")
        
        # You can implement your own parameter sampling logic here
        import random
        
        # Example: Random hyperparameter selection
        epochs_options = [50, 100, 200, 300]
        learning_rate_options = [0.001, 0.005, 0.01, 0.05]
        
        selected_epochs = random.choice(epochs_options)
        selected_lr = random.choice(learning_rate_options)
        
        set_component_param(modified_config, 'DIETClassifier', 'epochs', selected_epochs)
        set_component_param(modified_config, 'DIETClassifier', 'learning_rate', selected_lr)
        
        logger.info(f"Selected epochs: {selected_epochs}, learning_rate: {selected_lr}")
    
    # Example 3: Environment variable based configuration
    import os
    if os.getenv('RASA_EXPERIMENT_MODE') == 'fast':
        logger.info("Fast experiment mode detected")
        set_component_param(modified_config, 'DIETClassifier', 'epochs', 10)
        set_component_param(modified_config, 'DIETClassifier', 'eval_every_number_of_epochs', 2)
    
    return modified_config


def set_component_param(config: Dict[Text, Any], component_name: str, param_path: str, value: Any) -> None:
    """Set a parameter for a specific pipeline component.
    
    Args:
        config: Configuration dictionary to modify
        component_name: Name of the component (e.g., 'DIETClassifier')
        param_path: Parameter path (e.g., 'epochs' or 'early_stopping.patience')
        value: Value to set
    """
    pipeline = config.get('pipeline', [])
    
    for component in pipeline:
        if isinstance(component, dict) and component.get('name') == component_name:
            if '.' in param_path:
                # Handle nested parameters like 'early_stopping.patience'
                parts = param_path.split('.')
                current = component
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                component[param_path] = value
            
            logger.info(f"Set {component_name}.{param_path} = {value}")
            return
    
    logger.warning(f"Component {component_name} not found in pipeline")


def set_policy_param(config: Dict[Text, Any], policy_name: str, param_path: str, value: Any) -> None:
    """Set a parameter for a specific policy.
    
    Args:
        config: Configuration dictionary to modify
        policy_name: Name of the policy (e.g., 'UnexpecTEDIntentPolicy')
        param_path: Parameter path (e.g., 'epochs')
        value: Value to set
    """
    policies = config.get('policies', [])
    
    for policy in policies:
        if isinstance(policy, dict) and policy.get('name') == policy_name:
            if '.' in param_path:
                # Handle nested parameters
                parts = param_path.split('.')
                current = policy
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                policy[param_path] = value
            
            logger.info(f"Set {policy_name}.{param_path} = {value}")
            return
    
    logger.warning(f"Policy {policy_name} not found in policies")


# Example function for advanced use cases
def get_optimal_batch_size_for_dataset_size(dataset_size: int) -> list:
    """Calculate optimal batch sizes based on dataset size.
    
    Args:
        dataset_size: Number of training examples
        
    Returns:
        List of [initial_batch_size, final_batch_size]
    """
    if dataset_size < 100:
        return [16, 32]
    elif dataset_size < 1000:
        return [32, 128]
    elif dataset_size < 10000:
        return [64, 256]
    else:
        return [128, 512]
'''


def save_sweep_script_template(output_path: Text) -> None:
    """Save a template sweep script to a Python file.
    
    Args:
        output_path: Path where to save the template.
    """
    template_content = create_sweep_script_template()
    
    try:
        with open(output_path, 'w') as f:
            f.write(template_content)
        logger.info(f"Saved sweep script template to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save sweep script template: {e}")