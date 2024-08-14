import yaml


class Config:
    def __init__(self, config=None):
        if config is not None:
            for key, value in config.items():
                setattr(self, key, value)


def parse_run_config(config_path):
    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return Config(config)