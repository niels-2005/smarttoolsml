import yaml


def get_yaml_params(filepath: str):
    params = yaml.safe_load(filepath)["preprocess"]
    return params
