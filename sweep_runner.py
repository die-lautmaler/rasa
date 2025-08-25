"""
Wandb Sweep Runner for Rasa

This script demonstrates how to set up and run wandb sweeps with Rasa training.
It provides a complete example of hyperparameter optimization workflow.

Usage:
1. First, create a sweep: python sweep_runner.py create-sweep
2. Then, run the sweep: python sweep_runner.py run-sweep SWEEP_ID
3. Or run both: python sweep_runner.py auto

Requirements:
- wandb installed and logged in
- Rasa project with config.yaml and training data
"""

import argparse
import logging
import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    logger.error("wandb is not installed. Install with: pip install wandb")
    WANDB_AVAILABLE = False
    sys.exit(1)


class RasaSweepRunner:
    """Manages wandb sweeps for Rasa training."""
    
    def __init__(self, project_name: str = "rasa-hyperparameter-optimization"):
        """Initialize the sweep runner.
        
        Args:
            project_name: Name of the wandb project.
        """
        self.project_name = project_name
        self.sweep_config_file = "example_sweep_config.yaml"
        self.sweep_script_file = "example_sweep_script.py"
        self.config_file = "config.yml"
        self.nlu_data = "data"
        
    def create_sweep_config(self) -> Dict[str, Any]:
        """Create a sweep configuration.
        
        Returns:
            Sweep configuration dictionary.
        """
        return {
            "project": self.project_name,
            "method": "bayes",
            "metric": {
                "goal": "minimize",
                "name": "validation/total_loss"
            },
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 10
            },
            "parameters": {
                # DIET Classifier core parameters
                "diet_epochs": {"values": [50, 100, 200]},
                "diet_learning_rate": {
                    "min": 0.0001,
                    "max": 0.01,
                    "distribution": "log_uniform_values"
                },
                "diet_batch_size": {"values": [[32, 128], [64, 256]]},
                "diet_transformer_size": {"values": [64, 128, 256]},
                "diet_num_transformer_layers": {"values": [1, 2, 3]},
                "diet_drop_rate": {
                    "min": 0.0,
                    "max": 0.5,
                    "distribution": "uniform"
                },
                
                # Early stopping parameters
                "early_stopping_patience": {"values": [5, 10, 15]},
                "early_stopping_monitor": {"values": ["val_loss", "val_i_acc"]},
                "early_stopping_min_delta": {
                    "min": 0.0001,
                    "max": 0.01,
                    "distribution": "log_uniform_values"
                }
            }
        }
    
    def create_sweep(self, sweep_config: Optional[Dict[str, Any]] = None) -> str:
        """Create a wandb sweep.
        
        Args:
            sweep_config: Optional custom sweep configuration.
            
        Returns:
            Sweep ID.
        """
        if sweep_config is None:
            if Path(self.sweep_config_file).exists():
                logger.info(f"Loading sweep config from {self.sweep_config_file}")
                with open(self.sweep_config_file, 'r') as f:
                    sweep_config = yaml.safe_load(f)
            else:
                logger.info("Using default sweep configuration")
                sweep_config = self.create_sweep_config()
        
        # Create the sweep
        sweep_id = wandb.sweep(sweep=sweep_config, project=self.project_name)
        logger.info(f"Created sweep with ID: {sweep_id}")
        logger.info(f"View sweep at: https://wandb.ai/{wandb.api.default_entity}/{self.project_name}/sweeps/{sweep_id}")
        
        return sweep_id
    
    def run_sweep_iteration(self):
        """Run a single sweep iteration (called by wandb agent)."""
        logger.info("Starting sweep iteration")
        
        # Initialize wandb run
        wandb.init()
        
        # Build the training command
        cmd = [
            "rasa", "train", "nlu",
            "--config", self.config_file,
            "--nlu", self.nlu_data,
            "--out", "models",
            "--wandb",
            "--sweep-script", self.sweep_script_file
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run Rasa training
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Training completed successfully")
            logger.debug(f"Training output: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed with return code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            logger.error(f"Standard output: {e.stdout}")
            
            # Log the error to wandb
            wandb.log({"training_error": 1, "error_message": str(e)})
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error during training: {e}")
            wandb.log({"training_error": 1, "error_message": str(e)})
            raise
        
        finally:
            # Finish the wandb run
            wandb.finish()
    
    def run_sweep_agent(self, sweep_id: str, count: Optional[int] = None):
        """Run a sweep agent.
        
        Args:
            sweep_id: The wandb sweep ID.
            count: Number of runs to execute (None for unlimited).
        """
        logger.info(f"Starting sweep agent for sweep: {sweep_id}")
        
        if count:
            logger.info(f"Will run {count} iterations")
        else:
            logger.info("Running indefinitely (Ctrl+C to stop)")
            
        # Run the sweep agent
        wandb.agent(sweep_id, function=self.run_sweep_iteration, count=count)
    
    def validate_setup(self) -> bool:
        """Validate that all required files exist.
        
        Returns:
            True if setup is valid, False otherwise.
        """
        required_files = [self.config_file, self.sweep_script_file]
        required_dirs = [self.nlu_data]
        
        missing_items = []
        
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_items.append(f"File: {file_path}")
                
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_items.append(f"Directory: {dir_path}")
        
        if missing_items:
            logger.error("Missing required files/directories:")
            for item in missing_items:
                logger.error(f"  - {item}")
            return False
            
        logger.info("Setup validation passed")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rasa Wandb Sweep Runner")
    parser.add_argument("action", choices=["create-sweep", "run-sweep", "auto", "validate"],
                       help="Action to perform")
    parser.add_argument("--sweep-id", type=str, help="Sweep ID (for run-sweep action)")
    parser.add_argument("--count", type=int, help="Number of sweep runs to execute")
    parser.add_argument("--project", type=str, default="rasa-hyperparameter-optimization",
                       help="Wandb project name")
    
    args = parser.parse_args()
    
    if not WANDB_AVAILABLE:
        logger.error("wandb is not available")
        sys.exit(1)
    
    runner = RasaSweepRunner(project_name=args.project)
    
    if args.action == "validate":
        if runner.validate_setup():
            logger.info("Setup is valid, ready to run sweeps!")
        else:
            sys.exit(1)
            
    elif args.action == "create-sweep":
        if not runner.validate_setup():
            sys.exit(1)
        
        sweep_id = runner.create_sweep()
        print(f"Sweep created with ID: {sweep_id}")
        print(f"Run with: python {__file__} run-sweep --sweep-id {sweep_id}")
        
    elif args.action == "run-sweep":
        if not args.sweep_id:
            logger.error("--sweep-id is required for run-sweep action")
            sys.exit(1)
            
        if not runner.validate_setup():
            sys.exit(1)
            
        runner.run_sweep_agent(args.sweep_id, count=args.count)
        
    elif args.action == "auto":
        if not runner.validate_setup():
            sys.exit(1)
            
        # Create sweep and run agent
        sweep_id = runner.create_sweep()
        logger.info(f"Created sweep: {sweep_id}")
        logger.info("Starting sweep agent...")
        runner.run_sweep_agent(sweep_id, count=args.count)


if __name__ == "__main__":
    main()