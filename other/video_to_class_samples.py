from moviepy.editor import VideoFileClip
from os.path        import splitext, exists
from scipy.misc     import imsave
from os             import makedirs
from sys            import argv
import cv2
       

video_file = argv[1] 
class_name = splitext(video_file)[0]
class_name = ''.join(i for i in class_name if not i.isdigit())
i, cascade = 0, cv2.CascadeClassifier('../models/cat_lbp.xml')

if not exists('../data/positive/{}'.format(class_name)):
    makedirs('../data/positive/{}'.format(class_name))

def detect(image):
    global i
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(gray)
            for face in faces:
                x,y,w,h = face
                image   = image[y:y+h,x:x+w]
                if image.shape[0]>63 and image.shape[1]>63:
                    save_name = '../data/positive/{}/{}.png'.format(class_name, i)
                    imsave(save_name, image)
                    i += 1

for frame in VideoFileClip(video_file).iter_frames():
    detect(frame)

