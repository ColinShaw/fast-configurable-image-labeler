import cv2


class RealtimeTest(object):

    def __init__(self):
        self.__camera = cv2.VideoCapture(0)

    def test(self):
        while True:
            ret, image = self.__camera.read()
            if ret:
                cv2.imshow('Video Test', image)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        self.__camera.release()
        cv2.destroyAllWindows()

