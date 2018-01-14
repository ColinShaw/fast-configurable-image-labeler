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


CONFIDENCE_THRESHOLD = 0.7
PCA_COMPONENTS       = 30
NULL_CLASSNAME       = 'null'
CAFFE_PROTOTYPE      = 'models/mobilenet_ssd.prototxt'
CAFFE_MODEL          = 'models/mobilenet_ssd.caffemodel'
CLASSIFIER_MODEL     = 'models/classifier.p'
DECOMPOSITION_MODEL  = 'models/decomposition.p'


class ImageDetect(object):

    def __init__(self):
        self.__make_convolutional_model()
        self.__make_caffe_model()
        if isfile(CLASSIFIER_MODEL) and isfile(DECOMPOSITION_MODEL):
            print('Using existing models...')
            self.__load_classifier()
            self.__load_decomposition()
        else:
            print('Training...')
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

    def __make_caffe_model(self):
        self.__caffe = cv2.dnn.readNetFromCaffe(CAFFE_PROTOTYPE, CAFFE_MODEL)

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
        image = imresize(image, (128,128)).astype(np.float32)
        image = np.reshape(image, (1,128,128,3))
        label = self.__model.predict([image])
        return label.ravel()

    def __generate_training_data(self):
        features, labels = [], []
        dirlist = listdir('data')
        dir_names = [d for d in dirlist if isdir('data/{}'.format(d))]
        for dir_name in dir_names:
            dirlist = listdir('data/{}'.format(dir_name))
            image_names = [f for f in dirlist if 'png' in f]
            for image_name in image_names:
                labels.append(dir_name)
                feature = imread('data/{}/{}'.format(dir_name, image_name))
                feature = self.__conv_predict(feature)
                features.append(feature)
        return features, labels

    def __train_classifier(self):
        features, labels = self.__generate_training_data()
        self.__decomposition = PCA(n_components=PCA_COMPONENTS)
        features = self.__decomposition.fit_transform(features)
        self.__classifier = LinearSVC()
        self.__classifier.fit(features, labels)
   
    def annotations(self, image):
        results = []
        h,w     = image.shape[:2]
        resized = cv2.resize(image, (300,300))
        scaled  = cv2.dnn.blobFromImage(
            resized,
            0.007843,
            (300,300), 
            127.5
        )
        self.__caffe.setInput(scaled)
        cats = self.__caffe.forward()
        for i in np.arange(0, cats.shape[2]):
            if cats[0,0,i,2] > CONFIDENCE_THRESHOLD and cats[0,0,i,1] == 8:
                bounds  = cats[0,0,i,3:7] * np.array([w,h,w,h])
                bounds  = bounds.astype(np.int)
                a,b,c,d = bounds
                image   = image[a:c,b:d]
                if image.shape[0]>127 and image.shape[1]>127:
                    feature = self.__conv_predict(image)
                    feature = self.__decomposition.transform([feature])
                    label   = self.__classifier.predict(feature)[0]
                    results.append((bounds, label))
        return results 

    def label_image(self, image):
        items = self.annotations(image)
        for item in items:
            #if item[1] != NULL_CLASSNAME:
            if True:
                cv2.rectangle(
                    image,
                    (item[0][0],item[0][1]),
                    (item[0][2],item[0][3]),
                    (0,255,0),
                    1
                )
                cv2.putText(
                    image,
                    item[1],
                    (item[0][0],item[0][1]-2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0)
                )
        return image

