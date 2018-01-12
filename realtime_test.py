import cv2

camera  = cv2.VideoCapture(0)
while True:
    ret, image = camera.read()
    if ret == True:
        cv2.imshow('Video Test', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()

