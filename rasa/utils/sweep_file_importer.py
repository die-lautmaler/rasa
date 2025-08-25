"""Custom file importer for sweep runs that supports config modifications."""

import logging
from typing import Dict, List, Optional, Text, Union

from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.importers.rasa import RasaFileImporter

logger = logging.getLogger(__name__)


class SweepAwareFileImporter(TrainingDataImporter):
    """File importer that can use modified configurations for sweep runs."""

    def __init__(
        self,
        config_file: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[Union[List[Text], Text]] = None,
        modified_config: Optional[Dict] = None,
    ):
        """Initialize the sweep-aware file importer.
        
        Args:
            config_file: Path to the original config file.
            domain_path: Path to the domain file.
            training_data_paths: Paths to training data.
            modified_config: Modified configuration to use instead of file config.
        """
        self._base_importer = RasaFileImporter(
            config_file=config_file,
            domain_path=domain_path,
            training_data_paths=training_data_paths
        )
        self._modified_config = modified_config
        
        logger.info(f"SweepAwareFileImporter initialized with modified_config: {modified_config is not None}")

    def get_config(self) -> Dict:
        """Get configuration - uses modified config if provided, otherwise original."""
        if self._modified_config is not None:
            logger.debug("Using modified configuration from sweep")
            return self._modified_config
        else:
            logger.debug("Using original configuration from file")
            return self._base_importer.get_config()

    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Get config file path."""
        return self._base_importer.get_config_file_for_auto_config()

    def get_domain(self):
        """Get domain."""
        return self._base_importer.get_domain()

    def get_stories(self, exclusion_percentage: Optional[int] = None):
        """Get stories."""
        return self._base_importer.get_stories(exclusion_percentage)

    def get_conversation_tests(self):
        """Get conversation tests."""
        return self._base_importer.get_conversation_tests()

    def get_nlu_data(self, language: Optional[Text] = "en"):
        """Get NLU data."""
        return self._base_importer.get_nlu_data(language)

    def update_config(self, new_config: Dict) -> None:
        """Update the configuration used by this importer.
        
        Args:
            new_config: New configuration to use.
        """
        logger.info("Updating configuration for sweep run")
        self._modified_config = new_config