"""
Example Rasa Sweep Script for Hyperparameter Optimization

This script demonstrates how to use wandb sweep parameters with Rasa training.
It can be used with: rasa train nlu --wandb --sweep-script example_sweep_script.py

The script integrates with wandb sweeps and modifies the Rasa configuration
based on the sweep parameters defined in your sweep config.
"""

import logging
from typing import Dict, Text, Any

logger = logging.getLogger(__name__)

# Import wandb for sweep integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def modify_config_for_sweep(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """Modify the Rasa config based on wandb sweep parameters.
    
    This function is called before training starts and allows you to modify
    any part of the Rasa configuration based on sweep parameters.
    
    Args:
        config: Original Rasa configuration dictionary
        
    Returns:
        Modified configuration dictionary
    """
    import copy
    modified_config = copy.deepcopy(config)
    
    logger.info("Applying wandb sweep parameters to Rasa config")
    
    if not WANDB_AVAILABLE or not wandb.config:
        logger.warning("wandb not available or no sweep parameters found")
        return modified_config
    
    # Log all available sweep parameters
    sweep_params = dict(wandb.config)
    logger.info(f"Available sweep parameters: {list(sweep_params.keys())}")
    
    # Apply DIET Classifier parameters
    diet_params_applied = []
    
    if hasattr(wandb.config, 'diet_epochs'):
        set_component_param(modified_config, 'DIETClassifier', 'epochs', wandb.config.diet_epochs)
        diet_params_applied.append(f"epochs={wandb.config.diet_epochs}")
        
    if hasattr(wandb.config, 'diet_learning_rate'):
        set_component_param(modified_config, 'DIETClassifier', 'learning_rate', wandb.config.diet_learning_rate)
        diet_params_applied.append(f"learning_rate={wandb.config.diet_learning_rate}")
        
    if hasattr(wandb.config, 'diet_batch_size'):
        set_component_param(modified_config, 'DIETClassifier', 'batch_size', wandb.config.diet_batch_size)
        diet_params_applied.append(f"batch_size={wandb.config.diet_batch_size}")
        
    if hasattr(wandb.config, 'diet_transformer_size'):
        set_component_param(modified_config, 'DIETClassifier', 'transformer_size', wandb.config.diet_transformer_size)
        diet_params_applied.append(f"transformer_size={wandb.config.diet_transformer_size}")
        
    if hasattr(wandb.config, 'diet_num_transformer_layers'):
        set_component_param(modified_config, 'DIETClassifier', 'number_of_transformer_layers', wandb.config.diet_num_transformer_layers)
        diet_params_applied.append(f"number_of_transformer_layers={wandb.config.diet_num_transformer_layers}")
        
    if hasattr(wandb.config, 'diet_num_heads'):
        set_component_param(modified_config, 'DIETClassifier', 'number_of_attention_heads', wandb.config.diet_num_heads)
        diet_params_applied.append(f"number_of_attention_heads={wandb.config.diet_num_heads}")
        
    if hasattr(wandb.config, 'diet_drop_rate'):
        set_component_param(modified_config, 'DIETClassifier', 'drop_rate', wandb.config.diet_drop_rate)
        diet_params_applied.append(f"drop_rate={wandb.config.diet_drop_rate}")
        
    if hasattr(wandb.config, 'diet_embedding_dimension'):
        set_component_param(modified_config, 'DIETClassifier', 'embedding_dimension', wandb.config.diet_embedding_dimension)
        diet_params_applied.append(f"embedding_dimension={wandb.config.diet_embedding_dimension}")
        
    if hasattr(wandb.config, 'diet_regularization_constant'):
        set_component_param(modified_config, 'DIETClassifier', 'regularization_constant', wandb.config.diet_regularization_constant)
        diet_params_applied.append(f"regularization_constant={wandb.config.diet_regularization_constant}")
        
    if hasattr(wandb.config, 'diet_similarity_type'):
        set_component_param(modified_config, 'DIETClassifier', 'similarity_type', wandb.config.diet_similarity_type)
        diet_params_applied.append(f"similarity_type={wandb.config.diet_similarity_type}")
        
    if hasattr(wandb.config, 'diet_loss_type'):
        set_component_param(modified_config, 'DIETClassifier', 'loss_type', wandb.config.diet_loss_type)
        diet_params_applied.append(f"loss_type={wandb.config.diet_loss_type}")
        
    if hasattr(wandb.config, 'diet_num_neg'):
        set_component_param(modified_config, 'DIETClassifier', 'number_of_negative_examples', wandb.config.diet_num_neg)
        diet_params_applied.append(f"number_of_negative_examples={wandb.config.diet_num_neg}")
    
    # Apply Early Stopping parameters
    early_stopping_params_applied = []
    
    # First, ensure early stopping is enabled if we have sweep parameters for it
    if any(hasattr(wandb.config, param) for param in ['early_stopping_patience', 'early_stopping_monitor', 'early_stopping_min_delta']):
        set_component_param(modified_config, 'DIETClassifier', 'early_stopping.enabled', True)
        early_stopping_params_applied.append("enabled=True")
    
    if hasattr(wandb.config, 'early_stopping_patience'):
        set_component_param(modified_config, 'DIETClassifier', 'early_stopping.patience', wandb.config.early_stopping_patience)
        early_stopping_params_applied.append(f"patience={wandb.config.early_stopping_patience}")
        
    if hasattr(wandb.config, 'early_stopping_monitor'):
        set_component_param(modified_config, 'DIETClassifier', 'early_stopping.monitor', wandb.config.early_stopping_monitor)
        early_stopping_params_applied.append(f"monitor={wandb.config.early_stopping_monitor}")
        
    if hasattr(wandb.config, 'early_stopping_min_delta'):
        set_component_param(modified_config, 'DIETClassifier', 'early_stopping.min_delta', wandb.config.early_stopping_min_delta)
        early_stopping_params_applied.append(f"min_delta={wandb.config.early_stopping_min_delta}")
    
    # Apply CountVectorsFeaturizer parameters
    cv_params_applied = []
    
    if hasattr(wandb.config, 'count_vectors_min_ngram'):
        set_component_param(modified_config, 'CountVectorsFeaturizer', 'min_ngram', wandb.config.count_vectors_min_ngram)
        cv_params_applied.append(f"min_ngram={wandb.config.count_vectors_min_ngram}")
        
    if hasattr(wandb.config, 'count_vectors_max_ngram'):
        set_component_param(modified_config, 'CountVectorsFeaturizer', 'max_ngram', wandb.config.count_vectors_max_ngram)
        cv_params_applied.append(f"max_ngram={wandb.config.count_vectors_max_ngram}")
        
    if hasattr(wandb.config, 'count_vectors_max_features'):
        set_component_param(modified_config, 'CountVectorsFeaturizer', 'max_features', wandb.config.count_vectors_max_features)
        cv_params_applied.append(f"max_features={wandb.config.count_vectors_max_features}")
    
    # Log what was applied
    if diet_params_applied:
        logger.info(f"Applied DIETClassifier parameters: {', '.join(diet_params_applied)}")
        
    if early_stopping_params_applied:
        logger.info(f"Applied early stopping parameters: {', '.join(early_stopping_params_applied)}")
        
    if cv_params_applied:
        logger.info(f"Applied CountVectorsFeaturizer parameters: {', '.join(cv_params_applied)}")
    
    # Log the sweep configuration to wandb for easy tracking
    if WANDB_AVAILABLE:
        try:
            wandb.config.update({
                "diet_params_applied": diet_params_applied,
                "early_stopping_params_applied": early_stopping_params_applied,
                "cv_params_applied": cv_params_applied,
                "total_params_modified": len(diet_params_applied) + len(early_stopping_params_applied) + len(cv_params_applied)
            })
        except Exception as e:
            logger.warning(f"Failed to log sweep configuration to wandb: {e}")
    
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
            
            logger.debug(f"Set {component_name}.{param_path} = {value}")
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
            
            logger.debug(f"Set {policy_name}.{param_path} = {value}")
            return
    
    logger.warning(f"Policy {policy_name} not found in policies")