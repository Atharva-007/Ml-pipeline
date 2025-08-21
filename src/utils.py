import json
import logging
import os
from datetime import datetime

import yaml


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config file: {str(e)}")
        raise

def save_results(results, filename):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {filename}")

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

def create_directories(config):
    """Create necessary directories"""
    directories = [
        config['data']['raw_data_path'],
        config['data']['processed_data_path'],
        config['output']['model_path'],
        config['output']['results_path'],
        config['output']['plots_path'],
        "logs/"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")