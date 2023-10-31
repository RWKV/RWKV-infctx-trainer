#!/usr/bin/env python3
import sys
import yaml 
import os
from src.data import RWKVDataModule

# ----
# This script is used to preload the huggingface dataset
# that is configured in the config.yaml file
# ----

# Check for argument, else throw error
if len(sys.argv) < 2:
    print("No arguments supplied")
    print("Usage: python3 preload_datapath.py <config.yaml>")
    sys.exit(1)

# Check if the config file exists, else throw error (default assertion)
config_file = sys.argv[1]
assert os.path.exists(config_file), "Config file does not exist"

# Read the config file
with open(config_file, 'r') as f:
    lightning_config = yaml.safe_load(f)

# Check if the data is configured, else throw error (default assertion)
assert 'data' in lightning_config, "Data is not configured in the config file"

# Get the data object
data = lightning_config['data']

# Overwrite 'skip_datapath_setup' to False
data['skip_datapath_setup'] = False

# Run the preload data process
dataMod = RWKVDataModule(**data)
dataMod.prepare_data()