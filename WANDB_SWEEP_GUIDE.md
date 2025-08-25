# Wandb Sweep Integration for Rasa

This guide explains how to use Weights & Biases (wandb) sweeps for hyperparameter optimization in Rasa training.

## Overview

Wandb sweeps allow you to automatically search through different hyperparameter combinations to find the best configuration for your Rasa model. This integration provides:

- **Automatic sweep initialization** - just provide sweep config and script files
- **Flexible parameter modification** via Python sweep scripts  
- **Integration with wandb's optimization algorithms** (Bayesian, grid search, random search)
- **Automatic early stopping** based on validation metrics
- **One-command execution** - no manual sweep creation needed

## 🚀 Super Quick Start (New!)

### 1. Setup

Ensure you have wandb installed and logged in:

```bash
pip install wandb
wandb login
```

### 2. One-Command Auto Sweep

Create your sweep config and script, then run:

```bash
# Automatically creates and runs the wandb sweep!
rasa train nlu --sweep-script simple_sweep_example.py --sweep-config simple_sweep_config.yaml
```

That's it! The command will:
1. ✅ Load your sweep configuration
2. ✅ Create a wandb sweep automatically  
3. ✅ Run the sweep agent
4. ✅ Execute multiple training runs with different hyperparameters
5. ✅ Log everything to wandb for analysis

### 3. Alternative Usage Methods

```bash
# Traditional: Manual sweep script only (requires existing wandb sweep)
rasa train nlu --wandb --sweep-script example_sweep_script.py

# With custom log frequency
rasa train nlu --sweep-script my_sweep.py --sweep-config my_config.yaml --wandb-log-frequency 500

# Using the orchestration script (advanced)
python sweep_runner.py auto
```

## 📁 Simple Setup (Minimal Files)

For the auto-sweep feature, you only need these files:

```
your-rasa-project/
├── config.yml                    # Your existing Rasa config
├── data/                         # Your existing training data
├── simple_sweep_config.yaml      # Sweep configuration (NEW)
├── simple_sweep_example.py       # Parameter script (NEW)
└── models/                       # Output models
```

**Example `simple_sweep_config.yaml`:**
```yaml
project: my-rasa-optimization
method: bayes
metric:
  goal: minimize
  name: validation/total_loss
parameters:
  epochs: {values: [50, 100, 200]}
  learning_rate: {min: 0.001, max: 0.01, distribution: log_uniform_values}
  batch_size: {values: [[32, 128], [64, 256]]}
```

**Example `simple_sweep_example.py`:**
```python
def modify_config_for_sweep(config):
    import copy
    modified_config = copy.deepcopy(config)
    
    if wandb.config.epochs:
        set_component_param(modified_config, 'DIETClassifier', 'epochs', wandb.config.epochs)
    if wandb.config.learning_rate:
        set_component_param(modified_config, 'DIETClassifier', 'learning_rate', wandb.config.learning_rate)
    
    return modified_config

def set_component_param(config, component_name, param_path, value):
    # Helper function to set parameters (see full example for implementation)
    ...
```

Then just run:
```bash
rasa train nlu --sweep-script simple_sweep_example.py --sweep-config simple_sweep_config.yaml
```

## 📁 Advanced Setup (All Files)

For advanced usage with additional tools:

```
your-rasa-project/
├── config.yml                    # Your Rasa config
├── data/                         # Training data
├── example_sweep_config.yaml     # Wandb sweep configuration
├── example_sweep_script.py       # Parameter modification script
├── sweep_runner.py               # Sweep orchestration script
└── models/                       # Output models
```

## Sweep Script Format

Your sweep script must contain a `modify_config_for_sweep(config)` function:

```python
def modify_config_for_sweep(config: Dict[Text, Any]) -> Dict[Text, Any]:
    \"\"\"Modify config based on sweep parameters.\"\"\"
    import copy
    modified_config = copy.deepcopy(config)
    
    # Access wandb sweep parameters
    if wandb.config.diet_learning_rate:
        set_component_param(
            modified_config, 
            'DIETClassifier', 
            'learning_rate', 
            wandb.config.diet_learning_rate
        )
    
    return modified_config
```

## Available Parameters

### DIET Classifier Parameters

| Sweep Parameter | Rasa Config Path | Description |
|-----------------|------------------|-------------|
| `diet_epochs` | `epochs` | Number of training epochs |
| `diet_learning_rate` | `learning_rate` | Learning rate |
| `diet_batch_size` | `batch_size` | Batch size configuration |
| `diet_transformer_size` | `transformer_size` | Transformer hidden size |
| `diet_num_transformer_layers` | `number_of_transformer_layers` | Number of transformer layers |
| `diet_num_heads` | `number_of_attention_heads` | Number of attention heads |
| `diet_drop_rate` | `drop_rate` | Dropout rate |
| `diet_embedding_dimension` | `embedding_dimension` | Embedding dimension |
| `diet_regularization_constant` | `regularization_constant` | L2 regularization |
| `diet_similarity_type` | `similarity_type` | Similarity function type |
| `diet_loss_type` | `loss_type` | Loss function type |
| `diet_num_neg` | `number_of_negative_examples` | Negative sampling count |

### Early Stopping Parameters

| Sweep Parameter | Rasa Config Path | Description |
|-----------------|------------------|-------------|
| `early_stopping_patience` | `early_stopping.patience` | Epochs to wait for improvement |
| `early_stopping_monitor` | `early_stopping.monitor` | Metric to monitor |
| `early_stopping_min_delta` | `early_stopping.min_delta` | Minimum improvement threshold |

### CountVectorsFeaturizer Parameters

| Sweep Parameter | Rasa Config Path | Description |
|-----------------|------------------|-------------|
| `count_vectors_min_ngram` | `min_ngram` | Minimum n-gram size |
| `count_vectors_max_ngram` | `max_ngram` | Maximum n-gram size |
| `count_vectors_max_features` | `max_features` | Maximum feature count |

## Sweep Configuration

Example `sweep_config.yaml`:

```yaml
method: bayes
metric:
  goal: minimize
  name: validation/total_loss

parameters:
  diet_learning_rate:
    min: 0.0001
    max: 0.01
    distribution: log_uniform_values
  
  diet_epochs:
    values: [50, 100, 200, 300]
  
  early_stopping_patience:
    values: [5, 10, 15, 20]
```

## Optimization Strategies

### 1. Bayesian Optimization (Recommended)
```yaml
method: bayes
```
- Most efficient for expensive evaluations
- Good for continuous parameters
- Balances exploration vs exploitation

### 2. Grid Search
```yaml
method: grid
```
- Exhaustive search over discrete parameters
- Good for small parameter spaces
- Reproducible results

### 3. Random Search
```yaml
method: random
```
- Good baseline approach
- Works well with many parameters
- Easy to parallelize

## Metrics to Optimize

Common metrics to monitor:

| Metric | Description | Goal |
|--------|-------------|------|
| `validation/total_loss` | Overall validation loss | minimize |
| `validation/intent_loss` | Intent classification loss | minimize |
| `validation/entity_loss` | Entity recognition loss | minimize |
| `validation/intent_accuracy` | Intent accuracy | maximize |
| `validation/entity_f1_score` | Entity F1 score | maximize |

## Advanced Usage

### Custom Parameter Sampling

```python
def modify_config_for_sweep(config):
    import random
    
    # Custom sampling logic
    if random.random() < 0.5:
        learning_rate = random.uniform(0.001, 0.01)
    else:
        learning_rate = random.uniform(0.0001, 0.001)
    
    set_component_param(config, 'DIETClassifier', 'learning_rate', learning_rate)
    return config
```

### Environment-Based Configuration

```python
def modify_config_for_sweep(config):
    import os
    
    # Different configs based on environment
    if os.getenv('EXPERIMENT_MODE') == 'quick':
        set_component_param(config, 'DIETClassifier', 'epochs', 10)
    elif os.getenv('EXPERIMENT_MODE') == 'thorough':
        set_component_param(config, 'DIETClassifier', 'epochs', 500)
    
    return config
```

### Multi-Component Optimization

```python
def modify_config_for_sweep(config):
    # Optimize both DIET and TED policy
    if hasattr(wandb.config, 'diet_learning_rate'):
        set_component_param(config, 'DIETClassifier', 'learning_rate', wandb.config.diet_learning_rate)
    
    if hasattr(wandb.config, 'ted_epochs'):
        set_policy_param(config, 'UnexpecTEDIntentPolicy', 'epochs', wandb.config.ted_epochs)
    
    return config
```

## Best Practices

1. **Start Small**: Begin with a few key parameters before expanding
2. **Use Early Termination**: Configure wandb's early termination to save compute
3. **Monitor Resource Usage**: Set reasonable limits on epochs and model size
4. **Log Everything**: Include dataset characteristics and environment info
5. **Parallel Execution**: Run multiple agents for faster optimization

### Early Termination Configuration

```yaml
early_terminate:
  type: hyperband
  min_iter: 10
  eta: 3
  max_iter: 100
```

## Troubleshooting

### Common Issues

1. **"No sweep parameters found"**
   - Ensure wandb.init() is called before accessing wandb.config
   - Check that sweep was created properly

2. **"Component not found in pipeline"**
   - Verify component names match your config.yml exactly
   - Check for typos in component names

3. **"Failed to load sweep script"**
   - Ensure script path is correct
   - Check that script has required function

4. **Training fails during sweep**
   - Add error handling in your sweep script
   - Validate parameter ranges are reasonable

### Debug Mode

Add debug logging to your sweep script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

def modify_config_for_sweep(config):
    logger = logging.getLogger(__name__)
    logger.debug(f"Original config: {config}")
    
    # ... modifications ...
    
    logger.debug(f"Modified config: {modified_config}")
    return modified_config
```

## Examples

See the provided example files:
- `example_sweep_config.yaml` - Complete sweep configuration
- `example_sweep_script.py` - Full parameter modification script
- `sweep_runner.py` - Orchestration script for running sweeps

## Integration with CI/CD

```yaml
# .github/workflows/hyperparameter-search.yml
name: Hyperparameter Search
on:
  workflow_dispatch:
    inputs:
      sweep_count:
        description: 'Number of sweep runs'
        default: '10'

jobs:
  sweep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install rasa wandb
      - name: Run sweep
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: python sweep_runner.py auto --count ${{ github.event.inputs.sweep_count }}
```

This integration provides a flexible and powerful way to optimize your Rasa models using wandb's hyperparameter search capabilities!