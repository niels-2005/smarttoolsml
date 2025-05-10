import os

import yaml


def get_yaml_params(filepath: str):
    params = yaml.safe_load(filepath)["preprocess"]
    return params


def read_yaml_file(file_path: str) -> dict:

    with open(file_path, "rb") as yaml_file:
        return yaml.safe_load(yaml_file)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    with open(file_path, "w") as file:
        yaml.dump(content, file)
