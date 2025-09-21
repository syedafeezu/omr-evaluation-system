import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    """Configuration management for the OMR evaluation system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return self._get_default_config()
        else:
            # Create default config file
            default_config = self._get_default_config()
            self._save_config(default_config)
            return default_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'image_processing': {
                'max_image_width': 800,
                'noise_reduction': {
                    'bilateral_filter_d': 9,
                    'bilateral_filter_sigma_color': 75,
                    'bilateral_filter_sigma_space': 75
                },
                'contrast_enhancement': {
                    'clahe_clip_limit': 2.0,
                    'clahe_tile_grid_size': [8, 8]
                }
            },
            'bubble_detection': {
                'min_bubble_area': 50,
                'max_bubble_area': 500,
                'aspect_ratio_tolerance': 0.3,
                'fill_threshold': 0.3,
                'confidence_threshold': 0.7
            },
            'sheet_templates': {
                'default': {
                    'total_questions': 100,
                    'questions_per_row': 4,
                    'question_region': [100, 150, 600, 800],
                    'identifier_region': [50, 50, 200, 100]
                }
            },
            'database': {
                'type': 'sqlite',
                'path': 'data/omr_results.db',
                'backup_enabled': True,
                'backup_interval_hours': 24
            },
            'web_app': {
                'title': 'OMR Evaluation System',
                'port': 8501,
                'max_upload_size_mb': 50,
                'allowed_file_types': ['png', 'jpg', 'jpeg', 'pdf'],
                'results_per_page': 20
            },
            'scoring': {
                'subjects': [
                    'Mathematics',
                    'Physics', 
                    'Chemistry',
                    'Biology',
                    'English'
                ],
                'questions_per_subject': 20,
                'passing_threshold': 60
            },
            'export': {
                'formats': ['csv', 'xlsx', 'json'],
                'include_student_details': True,
                'include_answer_breakdown': True
            },
            'debug': {
                'save_intermediate_images': False,
                'log_level': 'INFO',
                'enable_profiling': False
            }
        }
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration key
            value: Value to set
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the final value
        config_ref[keys[-1]] = value
        
        # Save updated configuration
        self._save_config(self.config)
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()


# Global configuration instance
config = Config()

# Convenience functions for common configuration access
def get_image_processing_config() -> Dict[str, Any]:
    """Get image processing configuration."""
    return config.get('image_processing', {})

def get_bubble_detection_config() -> Dict[str, Any]:
    """Get bubble detection configuration."""
    return config.get('bubble_detection', {})

def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    return config.get('database', {})

def get_web_app_config() -> Dict[str, Any]:
    """Get web application configuration."""
    return config.get('web_app', {})

def get_scoring_config() -> Dict[str, Any]:
    """Get scoring configuration."""
    return config.get('scoring', {})

def get_export_config() -> Dict[str, Any]:
    """Get export configuration."""
    return config.get('export', {})

def get_debug_config() -> Dict[str, Any]:
    """Get debug configuration."""
    return config.get('debug', {})