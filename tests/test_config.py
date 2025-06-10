import os
import sys
import yaml

# Load config YAML and ensure key exists
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'config', 'agent_config.yaml')

def test_cargo_threshold_key():
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    assert 'mining' in cfg
    assert 'cargo_threshold_pct' in cfg['mining']
