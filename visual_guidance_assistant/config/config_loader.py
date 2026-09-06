"""Configuration loader from YAML file."""
import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load and validate configuration from YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    required_sections = ["model", "camera", "speech", "ui"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")

    config["camera"]["index"] = int(config["camera"]["index"])
    return config