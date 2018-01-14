from os import unlink
from config import Config


class Cleanup(object):

    def __init__(self):
        self.__config = Config()
        self.__files = [
            self.__config.get('classifier_model'),
            self.__config.get('decomposition_model')
        ]

    def cleanup(self):
        for f in self.__files:
            try:
                unlink(f)
            except:
                pass

