from os      import unlink, listdir, rmdir
from os.path import isfile, isdir
from config  import Config


class Cleanup(object):

    def __init__(self):
        self.__config = Config()
        self.__nc = self.__config.get('null_classname')
        self.__model_files = [
            self.__config.get('classifier_model'),
            self.__config.get('decomposition_model')
        ]

    def __unlink(self, path):
        try:
            unlink(path)
        except:
            pass

    def __rmdir(self, path):
        try:
            rmdir(path)
        except:
            pass

    def __unlink_models(self):
        for f in self.__model_files:
            self.__unlink(f)

    def __unlink_data(self):
        dirlist = listdir('data')
        dir_names = [d for d in dirlist if isdir('data/{}'.format(d))]
        dir_names = [d for d in dir_names if d != self.__nc]
        for dir_name in dir_names:
            dirlist = listdir('data/{}'.format(dir_name))
            image_names = [f for f in dirlist if 'png' in f]
            for image_name in image_names:
                self.__unlink('data/{}/{}'.format(dir_name, image_name))
            self.__rmdir('data/{}'.format(dir_name))

    def cleanup(self):
        self.__unlink_models()
        self.__unlink_data()

