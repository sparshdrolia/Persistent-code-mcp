"""
Configuration module for the persistent-code MCP server.

Handles loading and saving configuration settings for:
- LlamaIndex integration
- Embedding models
- Storage settings
- Logging levels
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Default configuration values
DEFAULT_CONFIG = {
    "llama_index": {
        "enabled": True,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 1024,
        "chunk_overlap": 20
    },
    "storage": {
        "use_sqlite": False,  # Whether to use SQLite for larger projects
        "compress_storage": False  # Whether to compress stored data
    },
    "logging": {
        "level": "INFO",
        "file_logging": False,
        "log_directory": "logs"
    },
    "advanced": {
        "max_tokens_per_component": 4000,
        "similarity_top_k": 5  # Number of similar components to retrieve
    }
}

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the persistent-code MCP server."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_dir: Directory for configuration files (default: ~/.persistent_code)
        """
        # Set config directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path.home() / ".persistent_code"
        
        # Create directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Config file path
        self.config_file = self.config_dir / "config.json"
        
        # Load or create config
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default.
        
        Returns:
            Configuration dictionary
        """
        # If config file exists, load it
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                
                # Update with any missing default values
                for section, values in DEFAULT_CONFIG.items():
                    if section not in config:
                        config[section] = values
                    else:
                        for key, value in values.items():
                            if key not in config[section]:
                                config[section][key] = value
                
                return config
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using default configuration.")
                return DEFAULT_CONFIG.copy()
        else:
            # Create default config
            logger.info(f"Creating default configuration at {self.config_file}")
            config = DEFAULT_CONFIG.copy()
            self.save_config(config)
            return config
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save (default: current config)
        """
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        try:
            return self.config[section][key]
        except KeyError:
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        # Create section if it doesn't exist
        if section not in self.config:
            self.config[section] = {}
        
        # Set value
        self.config[section][key] = value
        
        # Save config
        self.save_config()
    
    def get_embedding_model(self) -> str:
        """Get the configured embedding model.
        
        Returns:
            Embedding model name
        """
        return self.get("llama_index", "embedding_model", 
                        DEFAULT_CONFIG["llama_index"]["embedding_model"])
    
    def is_llama_index_enabled(self) -> bool:
        """Check if LlamaIndex integration is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return self.get("llama_index", "enabled", 
                        DEFAULT_CONFIG["llama_index"]["enabled"])
    
    def get_logging_level(self) -> str:
        """Get the configured logging level.
        
        Returns:
            Logging level as string
        """
        return self.get("logging", "level", 
                        DEFAULT_CONFIG["logging"]["level"])
    
    def get_similarity_top_k(self) -> int:
        """Get the number of similar components to retrieve.
        
        Returns:
            Number of similar components
        """
        return self.get("advanced", "similarity_top_k", 
                        DEFAULT_CONFIG["advanced"]["similarity_top_k"])
    
    def get_max_tokens_per_component(self) -> int:
        """Get the maximum number of tokens per component.
        
        Returns:
            Maximum tokens per component
        """
        return self.get("advanced", "max_tokens_per_component", 
                        DEFAULT_CONFIG["advanced"]["max_tokens_per_component"])

# Global configuration instance
config = Config()
