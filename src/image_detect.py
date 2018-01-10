from keras.applications    import mobilenet, imagenet_utils
from keras.models          import Model
from scipy.misc            import imread, imresize
from sklearn.svm           import LinearSVC
from sklearn.decomposition import PCA
from os.path               import isfile, splitext
from os                    import listdir
import numpy as np
import pickle
import cv2


CLASSIFIER_MODEL    = 'models/classifier.p'
DECOMPOSITION_MODEL = 'models/decomposition.p'
CASCADE_MODEL       = 'models/cat_hog.xml'


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

    def __vgg16_predict(self, image):
        image = imresize(image, (128,128)).astype(np.float32)
        image = imagenet_utils.preprocess_input(image)
        image = np.reshape(image, (1,128,128,3))
        label = self.__model.predict(image)
        return label.ravel()

    # Needs changing to directory labeling
    def __generate_training_data(self):
        features, labels = [], []
        dirlist = listdir('data/train')
        images  = [f for f in dirlist if 'jpg' in f] 
        for image in images:
            filename = splitext(image)[0]
            label = ''.join(i for i in filename if not i.isdigit())
            labels.append(label)
            feature = imread('data/train/{}.jpg'.format(filename))
            feature = self.__vgg16_predict(feature)
            features.append(feature)
        return features, labels

    def __train_classifier(self):
        print('Training...')
        features, labels = self.__generate_training_data()
        self.__decomposition = PCA()
        self.__classifier    = LinearSVC()
        self.__decomposition.fit(features)
        features = self.__decomposition.transform(features)
        self.__classifier.fit(features, labels)
   
    def detect(self, image):
        gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.__cascade.detectMultiScale(gray)
        for face in faces:
            x,y,w,h = face
            feature = self.__vgg16_predict(image[y:y+h,x:x+w])
            feature = self.__decomposition.transform([feature])
            label   = self.__classifier.predict(feature)[0]
            cv2.rectangle(
                image,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )
            cv2.putText(
                image,
                label,
                 (x+w+10,y+h),
                 0,
                 0.3,
                 (0,255,0)
            )
        return image

