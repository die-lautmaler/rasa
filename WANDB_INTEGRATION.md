# Weights & Biases Integration for Rasa NLU Training

This document describes the implementation of Weights & Biases (wandb) integration for the `rasa train nlu` command.

## Overview

The integration adds a `--wandb` flag to the `rasa train nlu` command that enables automatic logging of training metrics, model configuration, and trained model artifacts to Weights & Biases for experiment tracking and visualization.

## Features

- **Training Metrics Logging**: Automatically logs training data statistics, model configuration, and component-specific metrics
- **Component Timing**: Tracks training duration for each pipeline component
- **Model Artifacts**: Uploads trained model files as wandb artifacts
- **Configuration Tracking**: Logs all training parameters and model configuration
- **Error Handling**: Graceful fallback when wandb is not available or configured

## Usage

### Prerequisites

1. Install wandb as an optional dependency:
   ```bash
   pip install rasa[wandb]
   ```

2. Login to wandb (one-time setup):
   ```bash
   wandb login
   ```
   Or set the `WANDB_API_KEY` environment variable.

### Training with Wandb

Use the `--wandb` flag with your regular training command:

```bash
rasa train nlu --wandb
```

You can combine it with other flags:
```bash
rasa train nlu --wandb --config config.yml --nlu data/nlu.yml --out models/
```

## Logged Metrics

The integration automatically logs the following metrics:

### Training Data Metrics
- `num_training_examples`: Total number of training examples
- `num_intents`: Number of unique intents
- `num_entities`: Number of unique entities
- `num_entity_synonyms`: Number of entity synonyms
- `num_regex_features`: Number of regex features
- `num_lookup_tables`: Number of lookup tables

### Component Metrics
- `component_{name}_training_duration`: Training time for each component
- `component_{name}_completed`: Completion status for each component
- `component_{name}_config`: Configuration for each component

### Model Configuration
- Complete pipeline configuration
- Training parameters
- File paths and settings

### Artifacts
- Trained model file uploaded as wandb artifact

## Implementation Details

The integration consists of several components:

### 1. CLI Arguments (`rasa/cli/arguments/train.py`)
- Added `--wandb` flag to NLU training command
- Added `add_wandb_param()` function for argument parsing

### 2. Wandb Utilities (`rasa/utils/wandb_utils.py`)
- `WandBLogger`: Main class for wandb operations
- `create_wandb_logger()`: Factory function for logger creation
- `extract_training_metrics()`: Extracts metrics from training data
- Error handling for missing wandb installation

### 3. Training Integration (`rasa/model_training.py`)
- Modified `train_nlu()` to initialize wandb logging
- Extracts training metrics before training starts
- Logs model artifacts after successful training
- Ensures wandb run is properly finished

### 4. Training Hooks (`rasa/engine/training/hooks.py`)
- Added `WandBHook` for component-level metric logging
- Tracks component training duration
- Logs component configurations and completion status

### 5. Graph Trainer (`rasa/engine/training/graph_trainer.py`)
- Modified to accept wandb_logger parameter
- Integrates WandBHook into training pipeline

### 6. Dependencies (`pyproject.toml`)
- Added wandb as optional dependency
- Added to `wandb` and `full` extras

## Error Handling

The implementation includes comprehensive error handling:

- **Missing wandb**: Shows warning and continues training without logging
- **No API key**: Shows warning about wandb login requirement
- **Logging failures**: Logs warnings but doesn't interrupt training
- **Import failures**: Graceful fallback with debug messages

## Example Wandb Dashboard

When training with wandb enabled, you'll see:

1. **Overview Tab**: Run metadata and configuration
2. **Charts Tab**: Training metrics and component timings
3. **Artifacts Tab**: Trained model files
4. **Logs Tab**: Training logs and debug information

## Customization

You can customize the wandb integration by modifying:

- Project name in `WandBLogger` initialization
- Metric extraction logic in `extract_training_metrics()`
- Component-specific logging in `WandBHook`
- Run naming convention in `train_nlu()`

## Troubleshooting

### Common Issues

1. **"wandb not available" warning**:
   - Install wandb: `pip install rasa[wandb]`

2. **"W&B API key not found" warning**:
   - Run `wandb login` or set `WANDB_API_KEY` environment variable

3. **Training works but no wandb logging**:
   - Check that `--wandb` flag is used
   - Verify wandb installation and authentication

4. **Import errors**:
   - Ensure all dependencies are installed
   - Check Python environment

### Debug Mode

For debugging, set logging level to DEBUG:
```bash
export PYTHONPATH=. && python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
rasa train nlu --wandb
```

## Future Enhancements

Potential improvements for the integration:

1. **Cross-validation metrics**: Log CV scores if available
2. **Model comparison**: Compare multiple training runs
3. **Hyperparameter sweeps**: Integration with wandb sweeps
4. **Real-time metrics**: Stream metrics during training
5. **Evaluation metrics**: Log test set performance
6. **Custom dashboards**: Pre-built wandb dashboard templates