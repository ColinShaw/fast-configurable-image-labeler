from scipy.misc import imread, imsave
from os         import listdir
import numpy as np
import cv2


CONFIDENCE_THRESHOLD = 0.5
CAFFE_PROTOTYPE      = '../models/mobilenet_ssd.prototxt'
CAFFE_MODEL          = '../models/mobilenet_ssd.caffemodel'

global cnt
cnt, caffe = 0, cv2.dnn.readNetFromCaffe(CAFFE_PROTOTYPE, CAFFE_MODEL)

dirlist     = listdir('images/') 
image_names = [f for f in dirlist if 'jpg' in f] 

for image_name in image_names:
    image = imread('images/{}'.format(image_name))
    if len(image.shape) == 3 and image.shape[2] == 3:
        h, w    = image.shape[:2]
        resized = cv2.resize(image, (300,300))
        scaled  = cv2.dnn.blobFromImage(
            resized,
            0.007843,
            (300,300), 
            127.5
        )
        caffe.setInput(scaled)
        cats = caffe.forward()
        for i in np.arange(0, cats.shape[2]):
            if cats[0,0,i,2] > CONFIDENCE_THRESHOLD and cats[0,0,i,1] == 8:
                bounds  = cats[0,0,i,3:7] * np.array([w,h,w,h])
                a,b,c,d = bounds.astype(np.int)
                image   = image[a:c,b:d]
                if image.shape[0]>127 and image.shape[1]>127:
                    save_name = '../data/negative/{}.png'.format(cnt)
                    imsave(save_name, image)
                    cnt += 1

