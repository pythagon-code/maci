from ruamel.yaml import YAML
from types import SimpleNamespace


def _dict_to_namespace(obj: dict) -> SimpleNamespace:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_dict_to_namespace(item) for item in obj]
    else:
        return obj


def _get_config(filename: str) -> SimpleNamespace:
    yaml = YAML(typ="safe")

    with open(filename, "r") as f:
        data = yaml.load(f)

    if data is None:
        data = {}

    return _dict_to_namespace(data)


def run()