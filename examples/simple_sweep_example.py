"""
Simple Sweep Script Example - Auto Mode Compatible

This is a minimal sweep script that works with the new auto-sweep functionality.
Just run: rasa train nlu --sweep-script simple_sweep_example.py --sweep-config simple_sweep_config.yaml

The script will automatically create and run the wandb sweep!
"""

import logging
from typing import Dict, Text, Any

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def modify_config_for_sweep(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """Modify the Rasa config based on wandb sweep parameters."""
    import copy
    modified_config = copy.deepcopy(config)
    
    if not WANDB_AVAILABLE or not wandb.config:
        logger.info("No wandb sweep parameters found, using default config")
        return modified_config
    
    logger.info("Applying sweep parameters to config")
    
    # Simple parameter modifications
    params_applied = []
    
    # DIET Classifier epochs
    if hasattr(wandb.config, 'epochs'):
        set_component_param(modified_config, 'DIETClassifier', 'epochs', wandb.config.epochs)
        params_applied.append(f"epochs={wandb.config.epochs}")
    
    # DIET Classifier learning rate
    if hasattr(wandb.config, 'learning_rate'):
        set_component_param(modified_config, 'DIETClassifier', 'learning_rate', wandb.config.learning_rate)
        params_applied.append(f"learning_rate={wandb.config.learning_rate}")
    
    # Batch size
    if hasattr(wandb.config, 'batch_size'):
        set_component_param(modified_config, 'DIETClassifier', 'batch_size', wandb.config.batch_size)
        params_applied.append(f"batch_size={wandb.config.batch_size}")
    
    # Transformer size
    if hasattr(wandb.config, 'transformer_size'):
        set_component_param(modified_config, 'DIETClassifier', 'transformer_size', wandb.config.transformer_size)
        params_applied.append(f"transformer_size={wandb.config.transformer_size}")
    
    # Early stopping patience
    if hasattr(wandb.config, 'early_stopping_patience'):
        # Enable early stopping if patience is specified
        set_component_param(modified_config, 'DIETClassifier', 'early_stopping.enabled', True)
        set_component_param(modified_config, 'DIETClassifier', 'early_stopping.patience', wandb.config.early_stopping_patience)
        params_applied.append(f"early_stopping_patience={wandb.config.early_stopping_patience}")
    
    if params_applied:
        logger.info(f"Applied parameters: {', '.join(params_applied)}")
    else:
        logger.info("No sweep parameters found to apply")
    
    return modified_config


def set_component_param(config: Dict[Text, Any], component_name: str, param_path: str, value: Any) -> None:
    """Set a parameter for a specific pipeline component."""
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
            
            logger.debug(f"Set {component_name}.{param_path} = {value}")
            return
    
    logger.warning(f"Component {component_name} not found in pipeline")