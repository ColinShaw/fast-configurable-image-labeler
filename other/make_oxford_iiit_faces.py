from scipy.misc import imread, imsave
from os         import listdir
import csv
import cv2

cascade = cv2.CascadeClassifier('../models/cat_lbp.xml')
dirlist = listdir('images/') 
image_names = [f for f in dirlist if 'jpg' in f] 
for i, image_name in enumerate(image_names):
    image = imread('images/{}'.format(image_name))
    if len(image.shape) == 3:
        gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray)
        if len(faces) == 1:
            x,y,w,h = faces[0]
            image   = image[y:y+h,x:x+w]
            imsave('../data/negative/{}.png'.format(i), image)

