from moviepy.editor import load_video
from os.path        import splitext
from scipy.misc     import imsave
import cv2
import sys
       

video_file = argv[0] 
class_name = splitext(video_file)[0]
cascade    = cv2.CascadeClassifier(CASCADE_MODEL)
i          = 0

def detect(image):
    gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = self.__cascade.detectMultiScale(gray)
    if len(faces) == 1 and len(image[0][0]) == 3:
        x,y,w,h = faces[0]
        image   = image[y:y+h,x:x+w]
        imsave(image, 'data/{}/{}.png'.format(class_name, i))
        i += 1

clip = load_video(video_file)
for frame in clip.iter_frames():
    detect(frame)

