from moviepy.editor import load_video
from os.path        import splitext, exists
from scipy.misc     import imsave
from os             import makedirs
import cv2
import sys
       

video_file = argv[0] 
class_name = splitext(video_file)[0]
i, cascade = 0, cv2.CascadeClassifier('../models/cat_lbp.xml')
if not exists('../data/positive/{}'.format(class_name)):
    makedirs('../data/positive/{}'.format(class_name)

def detect(image):
    if len(image.shape) == 3:
        gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray)
        if len(faces) == 1:
            x,y,w,h = faces[0]
            image   = image[y:y+h,x:x+w]
            imsave('../data/positive/{}/{}.png'.format(class_name, i), image)
            i += 1

clip = load_video(video_file)
for frame in clip.iter_frames():
    detect(frame)

