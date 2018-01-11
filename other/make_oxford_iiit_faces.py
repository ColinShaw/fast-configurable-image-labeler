from scipy.misc import imread, imsave
import csv
import cv2

cascade = cv2.CascadeClassifier('../models/cat_lbp.xml')
with open('cats.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for i, name in enumerate(reader):
        image = imread('images/{}.jpg'.format(name[0]))
        gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray)
        if len(faces) == 1 and len(image[0][0]) == 3:
            x,y,w,h = faces[0]
            image   = image[y:y+h,x:x+w]
            imsave('../data/negative/{}.png'.format(i), image)

