import yaml
import ruamel.yaml


def load_cfg(cfg_path):
    with open(cfg_path) as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    return cfg