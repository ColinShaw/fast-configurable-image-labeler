from src.image_detect import ImageDetect
import cv2


img_det = ImageDetect()
camera  = cv2.VideoCapture(0)

while True:
    ret, image = camera.read()
    if ret == True:
        image = img_det.label_image(image)
        cv2.imshow('Cat Detection', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()

