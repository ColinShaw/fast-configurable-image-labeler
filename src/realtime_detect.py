from image_detect import ImageDetect
import cv2


class RealtimeDetect(object):

    def __init__(self):
        self.__imgdet = ImageDetect()
        self.__camera = cv2.VideoCapture(0)

    def detect(self):
        while True:
            ret, image = self.__camera.read()
            if ret:
                items = self.__imgdet.annotations(image)
                image = self.__imgdet.label_image(image, items)
                cv2.imshow('Image Detection', image)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        self.__camera.release()
        cv2.destroyAllWindows()

