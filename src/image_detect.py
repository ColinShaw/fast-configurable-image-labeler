from keras.applications    import mobilenet, imagenet_utils
from keras.models          import Model
from scipy.misc            import imread, imresize
from sklearn.svm           import LinearSVC
from sklearn.decomposition import PCA
from os.path               import isfile, splitext, isdir
from os                    import listdir
import numpy as np
import pickle
import cv2


CASCADE_MODEL       = 'models/cascades/cat_lbp.xml'
CLASSIFIER_MODEL    = 'models/trained/classifier.p'
DECOMPOSITION_MODEL = 'models/trained/decomposition.p'


class ImageDetect(object):

    def __init__(self):
        self.__make_convolutional_model()
        self.__make_cascade_model()
        if isfile(CLASSIFIER_MODEL) and isfile(DECOMPOSITION_MODEL):
            self.__load_classifier()
            self.__load_decomposition()
        else:
            self.__train_classifier()
            self.__save_classifier()
            self.__save_decomposition()

    def __make_convolutional_model(self):
        mn = mobilenet.MobileNet(
            weights     = 'imagenet', 
            include_top = False,
            input_shape = (128,128,3)
        )
        self.__model = Model(
            inputs  = [mn.input], 
            outputs = [mn.get_layer('conv_pw_13_relu').output]
        )

    def __make_cascade_model(self):
        self.__cascade = cv2.CascadeClassifier(CASCADE_MODEL)

    def __save_classifier(self):
        fp = open(CLASSIFIER_MODEL, 'wb')
        pickle.dump(self.__classifier, fp)

    def __load_classifier(self):
        fp = open(CLASSIFIER_MODEL, 'rb')
        self.__classifier = pickle.load(fp)

    def __save_decomposition(self):
        fp = open(DECOMPOSITION_MODEL, 'wb')
        pickle.dump(self.__decomposition, fp)

    def __load_decomposition(self):
        fp = open(DECOMPOSITION_MODEL, 'rb')
        self.__decomposition = pickle.load(fp)

    def __conv_predict(self, image):
        image = imresize(image, (128,128)).astype(np.float32)
        image = imagenet_utils.preprocess_input(image)
        image = np.reshape(image, (1,128,128,3))
        label = self.__model.predict(image)
        return label.ravel()

    def __generate_negative_training_data(self):
        features, labels = [], []
        dirlist = listdir('data/negative') 
        image_names = [f for f in dirlist if 'png' in f] 
        for image_name in image_names:
            labels.append('null')
            feature = imread('data/negative/{}'.format(image_name))
            feature = self.__conv_predict(feature)
            features.append(feature)
        return features, labels

    def __generate_positive_training_data(self):
        features, labels = [], []
        dirlist = listdir('data/positive')
        dir_names = [d for d in dirlist if isdir(d)]
        for dir_name in dir_names:
            dirlist = listdir('data/positive/{}'.format(dir_name))
            image_names = [f for f in dirlist if 'png' in f]
            for image_name in image_names:
                labels.append(dir_name)
                feature = imread('data/positive/{}/{}'.format(dir_name, image_name))
                feature = self.__conv_predict(feature)
                features.append(feature)
        return features, labels

    def __generate_training_data(self):
        features_p, labels_p = self.__generate_positive_training_data()
        features_n, labels_n = self.__generate_negative_training_data()
        return features_p + features_n, labels_p + labels_n

    def __train_classifier(self):
        print('Training...')
        features, labels = self.__generate_training_data()
        self.__decomposition = PCA()
        self.__classifier    = LinearSVC()
        self.__decomposition.fit(features)
        features = self.__decomposition.transform(features)
        self.__classifier.fit(features, labels)
   
    def annotations(self, image):
        results = []
        gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.__cascade.detectMultiScale(gray)
        for face in faces:
            x,y,w,h = face
            feature = self.__conv_predict(image[y:y+h,x:x+w])
            feature = self.__decomposition.transform([feature])
            label   = self.__classifier.predict(feature)[0]
            results.append((face, label))
        return results 

    def label_image(self, image):
        items = self.annotations(image)
        for item in items:
            x,y,w,h = item[0]
            cv2.putText(
                image,
                item[1],
                (x+w+10,y+h),
                0,
                0.3,
                (0,255,0)
            )
        return image

