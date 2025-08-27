"""Utilities for caching validation splits across sweep runs."""

import logging
import pickle
import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple, Any
import tempfile

from rasa.utils.tensorflow.model_data import RasaModelData

logger = logging.getLogger(__name__)


class ValidationSplitCache:
    """Cache for validation splits to ensure consistency across sweep runs."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the validation split cache.
        
        Args:
            cache_dir: Directory to store cache files. If None, uses temp directory.
        """
        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = Path(tempfile.gettempdir()) / "rasa_validation_cache"
        
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._current_split_id = None
    
    def _get_model_data_hash(self, model_data: RasaModelData, validation_split: float, random_seed: Optional[int]) -> str:
        """Generate a hash for the model data configuration.
        
        Args:
            model_data: The model data to hash.
            validation_split: The validation split fraction.
            random_seed: The random seed used for splitting.
            
        Returns:
            Hash string representing the configuration.
        """
        # Create hash based on data fingerprint, validation split, and random seed
        try:
            # Use a safer way to create data fingerprint
            signature = model_data.get_signature()
            signature_str = str(sorted(signature.keys())) if signature else "empty_signature"
            
            hash_components = [
                str(model_data.number_of_examples()),
                str(validation_split),
                str(random_seed or 0),
                signature_str,
                # Add additional fingerprint based on data structure
                str(len(str(model_data)))[:10]  # Simple size-based fingerprint
            ]
        except Exception as e:
            logger.warning(f"Failed to create detailed hash, using basic fingerprint: {e}")
            # Fallback to basic hash if signature fails
            hash_components = [
                str(model_data.number_of_examples()),
                str(validation_split), 
                str(random_seed or 0),
                "basic_fingerprint"
            ]
        
        hash_string = "_".join(hash_components)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _get_cache_path(self, data_hash: str) -> Path:
        """Get the cache file path for a given data hash.
        
        Args:
            data_hash: Hash of the data configuration.
            
        Returns:
            Path to the cache file.
        """
        return self._cache_dir / f"validation_split_{data_hash}.pkl"
    
    def has_cached_split(self, model_data: RasaModelData, validation_split: float, random_seed: Optional[int]) -> bool:
        """Check if a cached validation split exists for the given configuration.
        
        Args:
            model_data: The model data.
            validation_split: The validation split fraction.
            random_seed: The random seed used for splitting.
            
        Returns:
            True if a cached split exists, False otherwise.
        """
        if validation_split <= 0:
            return False
            
        try:
            total_examples = model_data.number_of_examples()
            data_hash = self._create_consistent_hash(total_examples, validation_split, random_seed)
            cache_path = self._get_cache_path(data_hash)
            return cache_path.exists()
        except Exception as e:
            logger.error(f"Error checking cached split: {e}")
            return False
    
    def cache_validation_split(
        self, 
        model_data: RasaModelData, 
        validation_data: RasaModelData,
        validation_split: float, 
        random_seed: Optional[int]
    ) -> str:
        """Cache a validation split.
        
        Args:
            model_data: The training model data after split.
            validation_data: The validation model data.
            validation_split: The validation split fraction.
            random_seed: The random seed used for splitting.
            
        Returns:
            The cache ID (hash) for this split.
        """
        if validation_split <= 0:
            return ""
            
        try:
            # First create a combined model data for hashing to ensure consistency
            total_examples = model_data.number_of_examples() + validation_data.number_of_examples()
            
            # Create a dummy model data object for consistent hashing
            # We use the training data as the base since it should be stable
            data_hash = self._create_consistent_hash(total_examples, validation_split, random_seed)
            cache_path = self._get_cache_path(data_hash)
            
            # Cache both the training and validation splits
            cache_data = {
                'training_data': model_data,
                'validation_data': validation_data,
                'validation_split': validation_split,
                'random_seed': random_seed,
                'total_examples': total_examples
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"Cached validation split: {data_hash}")
            self._current_split_id = data_hash
            return data_hash
            
        except Exception as e:
            logger.error(f"Failed to cache validation split: {e}")
            return ""
    
    def _create_consistent_hash(self, total_examples: int, validation_split: float, random_seed: Optional[int]) -> str:
        """Create a consistent hash for validation split based on basic parameters.
        
        This avoids issues with complex model data hashing.
        """
        hash_components = [
            str(total_examples),
            str(validation_split),
            str(random_seed or 0),
            "v2"  # Version marker for cache format
        ]
        
        hash_string = "_".join(hash_components)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def load_cached_split(
        self, 
        original_model_data: RasaModelData,
        validation_split: float, 
        random_seed: Optional[int]
    ) -> Optional[Tuple[RasaModelData, RasaModelData]]:
        """Load a cached validation split.
        
        Args:
            original_model_data: The original model data (before splitting).
            validation_split: The validation split fraction.
            random_seed: The random seed used for splitting.
            
        Returns:
            Tuple of (training_data, validation_data) if found, None otherwise.
        """
        if validation_split <= 0:
            return None
            
        try:
            # Use consistent hash approach for loading
            total_examples = original_model_data.number_of_examples()
            data_hash = self._create_consistent_hash(total_examples, validation_split, random_seed)
            cache_path = self._get_cache_path(data_hash)
            
            if not cache_path.exists():
                logger.debug(f"No cached validation split found for hash: {data_hash}")
                return None
                
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify the cached data is still valid
            cached_total = cache_data.get('total_examples', 0)
            current_total = original_model_data.number_of_examples()
            
            if cached_total != current_total:
                logger.warning(f"Cached validation split invalid due to data size change: {cached_total} != {current_total}")
                # Remove invalid cache file
                try:
                    cache_path.unlink()
                except Exception:
                    pass
                return None
            
            logger.info(f"Loaded cached validation split: {data_hash}")
            self._current_split_id = data_hash
            
            return cache_data['training_data'], cache_data['validation_data']
            
        except Exception as e:
            logger.error(f"Failed to load cached validation split: {e}")
            # If there's any error, fall back to no cached split
            return None
    
    def clear_cache(self) -> None:
        """Clear all cached validation splits."""
        try:
            for cache_file in self._cache_dir.glob("validation_split_*.pkl"):
                cache_file.unlink()
            logger.info("Cleared validation split cache")
        except Exception as e:
            logger.error(f"Failed to clear validation split cache: {e}")
    
    def prepare_for_new_sweep(self) -> None:
        """Prepare cache for a new sweep by clearing old cache entries.
        
        This should be called at the beginning of a new sweep to ensure
        fresh validation splits are generated for the new sweep.
        """
        self.clear_cache()
        self._current_split_id = None
        # Set environment variable to indicate we're in sweep mode
        os.environ['RASA_SWEEP_MODE'] = 'true'
        logger.info("Prepared validation split cache for new sweep")
    
    def enable_sweep_mode(self) -> None:
        """Manually enable sweep mode for validation split caching."""
        os.environ['RASA_SWEEP_MODE'] = 'true'
        logger.info("Enabled sweep mode for validation split caching")
    
    def disable_sweep_mode(self) -> None:
        """Disable sweep mode for validation split caching."""
        os.environ.pop('RASA_SWEEP_MODE', None)
        logger.info("Disabled sweep mode for validation split caching")
    
    def get_current_split_id(self) -> Optional[str]:
        """Get the current validation split ID.
        
        Returns:
            Current split ID if available, None otherwise.
        """
        return self._current_split_id
    
    def is_sweep_mode(self) -> bool:
        """Check if we're in sweep mode by looking for wandb environment variables.
        
        Returns:
            True if we're in a sweep context, False otherwise.
        """
        # Check for environment variables that indicate sweep mode
        env_indicators = bool(
            os.environ.get('WANDB_SWEEP_ID') or
            os.environ.get('WANDB_SWEEP_PARAM_PATH')
        )
        
        # Check for wandb sweep context
        wandb_indicators = self._check_wandb_sweep_context()
        
        # Also check if we're in an explicit sweep training session
        explicit_sweep = bool(os.environ.get('RASA_SWEEP_MODE', '').lower() == 'true')
        
        return env_indicators or wandb_indicators or explicit_sweep
    
    def _check_wandb_sweep_context(self) -> bool:
        """Check if we're in a wandb sweep context.
        
        Returns:
            True if in sweep context, False otherwise.
        """
        try:
            import wandb
            if wandb.run and hasattr(wandb.run, 'sweep_id') and wandb.run.sweep_id:
                return True
        except (ImportError, AttributeError):
            pass
        return False


# Global validation split cache instance
_validation_split_cache = ValidationSplitCache()


def get_validation_split_cache() -> ValidationSplitCache:
    """Get the global validation split cache instance.
    
    Returns:
        The global ValidationSplitCache instance.
    """
    return _validation_split_cache


def clear_validation_cache() -> None:
    """Clear the validation split cache.
    
    This is a convenience function for users to clear the cache manually.
    """
    cache = get_validation_split_cache()
    cache.clear_cache()


def prepare_for_sweep() -> None:
    """Prepare validation split cache for a new sweep.
    
    This is a convenience function that clears the cache and enables sweep mode.
    """
    cache = get_validation_split_cache()
    cache.prepare_for_new_sweep()