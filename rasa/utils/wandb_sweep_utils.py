"""Utilities for Weights & Biases Sweep integration with Rasa."""

import logging
import os
import yaml
import copy
from typing import Any, Dict, Optional, Text, List, Union
from pathlib import Path

from rasa.utils.wandb_utils import WANDB_AVAILABLE

logger = logging.getLogger(__name__)

# Import wandb if available
if WANDB_AVAILABLE:
    import wandb
else:
    wandb = None


class WandBSweepManager:
    """Manages wandb sweep configuration and parameter injection for Rasa training."""

    def __init__(self):
        """Initialize the sweep manager."""
        self.sweep_config = None
        self.sweep_id = None
        self.is_sweep_run = False

    def load_sweep_config(self, sweep_config_path: Text) -> Optional[Dict[Text, Any]]:
        """Load sweep configuration from YAML file.
        
        Args:
            sweep_config_path: Path to the sweep configuration YAML file.
            
        Returns:
            Sweep configuration dictionary or None if failed.
        """
        try:
            with open(sweep_config_path, 'r') as f:
                self.sweep_config = yaml.safe_load(f)
            
            logger.info(f"Loaded sweep configuration from {sweep_config_path}")
            return self.sweep_config
            
        except Exception as e:
            logger.error(f"Failed to load sweep configuration from {sweep_config_path}: {e}")
            return None

    def create_sweep(self, project_name: str = "rasa-sweep") -> Optional[str]:
        """Create a wandb sweep and return the sweep ID.
        
        Args:
            project_name: Name of the wandb project for the sweep.
            
        Returns:
            Sweep ID if successful, None otherwise.
        """
        if not WANDB_AVAILABLE or wandb is None:
            logger.error("wandb is not available. Install with 'pip install rasa[wandb]'")
            return None
            
        if self.sweep_config is None:
            logger.error("No sweep configuration loaded. Call load_sweep_config() first.")
            return None
            
        try:
            # Add project to sweep config if not present
            if 'project' not in self.sweep_config:
                self.sweep_config['project'] = project_name
                
            self.sweep_id = wandb.sweep(sweep=self.sweep_config, project=project_name)
            logger.info(f"Created wandb sweep with ID: {self.sweep_id}")
            return self.sweep_id
            
        except Exception as e:
            logger.error(f"Failed to create wandb sweep: {e}")
            return None

    def inject_sweep_parameters(self, base_config: Dict[Text, Any]) -> Dict[Text, Any]:
        """Inject wandb sweep parameters into Rasa config.
        
        Args:
            base_config: Base Rasa configuration dictionary.
            
        Returns:
            Modified configuration with sweep parameters injected.
        """
        if not WANDB_AVAILABLE or wandb is None:
            logger.warning("wandb not available, returning original config")
            return base_config
            
        # Check if this is a sweep run
        if not hasattr(wandb.config, 'keys') or len(wandb.config.keys()) == 0:
            logger.debug("No sweep parameters found, using base config")
            return base_config
            
        self.is_sweep_run = True
        config = copy.deepcopy(base_config)
        
        logger.info("Injecting wandb sweep parameters into Rasa config")
        
        # Map wandb sweep parameters to Rasa config paths
        param_mappings = self._get_parameter_mappings()
        
        for wandb_param, config_path in param_mappings.items():
            if hasattr(wandb.config, wandb_param):
                value = getattr(wandb.config, wandb_param)
                self._set_nested_config_value(config, config_path, value)
                logger.info(f"Set {config_path} = {value} (from sweep parameter {wandb_param})")
                
        return config

    def _get_parameter_mappings(self) -> Dict[Text, Text]:
        """Get mapping from wandb parameter names to Rasa config paths.
        
        Returns:
            Dictionary mapping wandb parameter names to config paths.
        """
        # Default parameter mappings - can be extended
        return {
            # DIET Classifier parameters
            'diet_epochs': 'pipeline.DIETClassifier.epochs',
            'diet_learning_rate': 'pipeline.DIETClassifier.learning_rate',
            'diet_batch_sizes': 'pipeline.DIETClassifier.batch_size',
            'diet_transformer_size': 'pipeline.DIETClassifier.transformer_size',
            'diet_num_transformer_layers': 'pipeline.DIETClassifier.number_of_transformer_layers',
            'diet_num_heads': 'pipeline.DIETClassifier.number_of_attention_heads',
            'diet_drop_rate': 'pipeline.DIETClassifier.drop_rate',
            'diet_embedding_dimension': 'pipeline.DIETClassifier.embedding_dimension',
            'diet_regularization_constant': 'pipeline.DIETClassifier.regularization_constant',
            'diet_similarity_type': 'pipeline.DIETClassifier.similarity_type',
            'diet_loss_type': 'pipeline.DIETClassifier.loss_type',
            'diet_num_neg': 'pipeline.DIETClassifier.number_of_negative_examples',
            
            # Early stopping parameters
            'early_stopping_enabled': 'pipeline.DIETClassifier.early_stopping.enabled',
            'early_stopping_monitor': 'pipeline.DIETClassifier.early_stopping.monitor',
            'early_stopping_patience': 'pipeline.DIETClassifier.early_stopping.patience',
            'early_stopping_min_delta': 'pipeline.DIETClassifier.early_stopping.min_delta',
            
            # TED Policy parameters  
            'ted_epochs': 'policies.UnexpecTEDIntentPolicy.epochs',
            'ted_learning_rate': 'policies.UnexpecTEDIntentPolicy.learning_rate',
            'ted_batch_sizes': 'policies.UnexpecTEDIntentPolicy.batch_size',
            'ted_transformer_size': 'policies.UnexpecTEDIntentPolicy.transformer_size',
            'ted_num_transformer_layers': 'policies.UnexpecTEDIntentPolicy.number_of_transformer_layers',
            'ted_drop_rate': 'policies.UnexpecTEDIntentPolicy.drop_rate',
            'ted_regularization_constant': 'policies.UnexpecTEDIntentPolicy.regularization_constant',
            
            # CountVectorsFeaturizer parameters
            'count_vectors_min_ngram': 'pipeline.CountVectorsFeaturizer.min_ngram',
            'count_vectors_max_ngram': 'pipeline.CountVectorsFeaturizer.max_ngram',
            'count_vectors_max_features': 'pipeline.CountVectorsFeaturizer.max_features',
        }

    def _set_nested_config_value(self, config: Dict[Text, Any], path: Text, value: Any) -> None:
        """Set a nested configuration value using dot notation path.
        
        Args:
            config: Configuration dictionary to modify.
            path: Dot-separated path to the configuration key.
            value: Value to set.
        """
        parts = path.split('.')
        current = config
        
        # Navigate to the target location
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # If we encounter a non-dict, we need to find the right component
                if isinstance(current[part], list):
                    # Handle pipeline/policies lists
                    component_found = False
                    for item in current[part]:
                        if isinstance(item, dict) and item.get('name') == parts[-2]:
                            current = item
                            component_found = True
                            break
                    if not component_found:
                        logger.warning(f"Could not find component {parts[-2]} in {part}")
                        return
                else:
                    logger.warning(f"Cannot set nested value at {path}: {part} is not a dict")
                    return
            else:
                current = current[part]
        
        # Handle special cases for pipeline and policies
        if parts[0] in ['pipeline', 'policies'] and len(parts) > 2:
            component_name = parts[1]
            param_name = parts[2]
            
            # Find the component in the list
            component_list = config.get(parts[0], [])
            for component in component_list:
                if isinstance(component, dict) and component.get('name') == component_name:
                    if len(parts) == 3:
                        component[param_name] = value
                    else:
                        # Handle nested parameters like early_stopping.enabled
                        nested_path = '.'.join(parts[2:])
                        self._set_nested_config_value(component, nested_path, value)
                    return
                    
            logger.warning(f"Could not find component {component_name} in {parts[0]}")
        else:
            # Set the final value
            current[parts[-1]] = value

    def is_sweep_active(self) -> bool:
        """Check if this run is part of a wandb sweep.
        
        Returns:
            True if this is a sweep run, False otherwise.
        """
        return self.is_sweep_run


def create_sweep_config_template() -> Dict[Text, Any]:
    """Create a template sweep configuration for Rasa.
    
    Returns:
        Template sweep configuration dictionary.
    """
    return {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'goal': 'minimize',
            'name': 'validation/total_loss'
        },
        'parameters': {
            # DIET Classifier hyperparameters
            'diet_epochs': {
                'values': [50, 100, 200, 300]
            },
            'diet_learning_rate': {
                'min': 0.0001,
                'max': 0.01,
                'distribution': 'log_uniform_values'
            },
            'diet_batch_sizes': {
                'values': [[32, 128], [64, 256], [128, 512]]
            },
            'diet_transformer_size': {
                'values': [64, 128, 256, 512]
            },
            'diet_num_transformer_layers': {
                'values': [1, 2, 3, 4]
            },
            'diet_num_heads': {
                'values': [2, 4, 8]
            },
            'diet_drop_rate': {
                'min': 0.0,
                'max': 0.5
            },
            'diet_embedding_dimension': {
                'values': [10, 20, 50, 100]
            },
            'diet_regularization_constant': {
                'min': 0.0001,
                'max': 0.01,
                'distribution': 'log_uniform_values'
            },
            'diet_similarity_type': {
                'values': ['auto', 'cosine', 'inner']
            },
            'diet_loss_type': {
                'values': ['cross_entropy', 'margin']
            },
            'diet_num_neg': {
                'values': [10, 20, 50, 100]
            },
            
            # Early stopping parameters
            'early_stopping_enabled': {
                'values': [True, False]
            },
            'early_stopping_monitor': {
                'values': ['val_loss', 'val_i_acc', 'val_e_f1']
            },
            'early_stopping_patience': {
                'values': [5, 10, 15, 20]
            },
            'early_stopping_min_delta': {
                'min': 0.0001,
                'max': 0.01
            },
            
            # CountVectorsFeaturizer parameters
            'count_vectors_min_ngram': {
                'values': [1, 2]
            },
            'count_vectors_max_ngram': {
                'values': [2, 3, 4, 5]
            },
            'count_vectors_max_features': {
                'values': [1000, 5000, 10000, 50000]
            }
        }
    }


def save_sweep_config_template(output_path: Text) -> None:
    """Save a template sweep configuration to a YAML file.
    
    Args:
        output_path: Path where to save the template.
    """
    template = create_sweep_config_template()
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        logger.info(f"Saved sweep configuration template to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save sweep configuration template: {e}")


def run_sweep_agent(sweep_id: str, function_to_run, count: Optional[int] = None) -> None:
    """Run a wandb sweep agent.
    
    Args:
        sweep_id: The wandb sweep ID.
        function_to_run: Function to execute for each sweep run.
        count: Number of runs to execute (None for unlimited).
    """
    if not WANDB_AVAILABLE or wandb is None:
        logger.error("wandb is not available. Install with 'pip install rasa[wandb]'")
        return
        
    try:
        wandb.agent(sweep_id, function=function_to_run, count=count)
    except Exception as e:
        logger.error(f"Failed to run sweep agent: {e}")


# Global sweep manager instance
_sweep_manager = WandBSweepManager()


def get_sweep_manager() -> WandBSweepManager:
    """Get the global sweep manager instance.
    
    Returns:
        The global WandBSweepManager instance.
    """
    return _sweep_manager