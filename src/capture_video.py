from moviepy.editor import ImageSequenceClip
import cv2


class CaptureVideo(object):

    def __init__(self, filename):
        self.__filename = 'videos/{}'.format(filename)
        self.__camera   = cv2.VideoCapture(0)

    def capture(self):
        frames = []
        while True:
            ret, image = self.__camera.read()
            if ret:
                cv2.imshow('Video Capture', image)
                frames.append(image)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        self.__camera.release()
        cv2.destroyAllWindows()
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(self.__filename, audio=False)

