from scipy.misc import imread, imsave
from os         import listdir
import cv2


dirlist     = listdir('images/') 
image_names = [f for f in dirlist if 'jpg' in f] 
i, cascade  = 0, cv2.CascadeClassifier('../models/cat_lbp.xml')

for image_name in image_names:
    image = imread('images/{}'.format(image_name))
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(gray)
            for face in faces:
                x,y,w,h = face
                image   = image[y:y+h,x:x+w]
                if image.shape[0]>63 and image.shape[1]>63:
                    save_name = '../data/negative/{}.png'.format(i)
                    imsave(save_name, image)
                    i += 1

