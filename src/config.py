import yaml


class Config(object):

    def __init__(self):
        with open('config.yaml') as f:
            self.__config = yaml.safe_load(f)

    def get(self, param):
        return self.__config[param]

