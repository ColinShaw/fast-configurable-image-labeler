from moviepy.editor import VideoFileClip
from os.path        import exists, splitext
from scipy.misc     import imsave
from os             import makedirs, listdir
import cv2
       

i, cascade = 0, cv2.CascadeClassifier('../models/cat_lbp.xml')

def detect(class_name, image):
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

dirlist = listdir('videos')
video_names = [f for f in dirlist if 'mp4' in f]
for video_name in video_names:
    class_name = splitext(video_name)[0]
    class_name = ''.join(i for i in class_name if not i.isdigit())

    if not exists('../data/positive/{}'.format(class_name)):
        makedirs('../data/positive/{}'.format(class_name))

    video_file = 'videos/{}'.format(video_name)
    for frame in VideoFileClip(video_file).iter_frames():
        detect(class_name, frame)

