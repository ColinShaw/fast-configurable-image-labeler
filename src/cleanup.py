from os import unlink


CLASSIFIER_MODEL     = 'models/classifier.p'
DECOMPOSITION_MODEL  = 'models/decomposition.p'


class Cleanup(object):

    def __init__(self):
        self.__files = [CLASSIFIER_MODEL, DECOMPOSITION_MODEL]

    def cleanup(self):
        for f in self.__files:
            try:
                unlink(f)
            except:
                pass

