"""funcproc.core.config
====================
Load project / analysis YAML configs (e.g. configs/example_funcproc_config.yml
for paths & denoise defaults, configs/example_prf_config.yml for pRF settings).

Commands currently take explicit paths/flags; this loader is the seam for
wiring in config-file defaults as the package grows.
"""

import os


def load_config(path):
    """Read a YAML config file into a dict (empty dict if the file is empty)."""
    import yaml
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}