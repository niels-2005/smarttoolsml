import cv2


def resize_img(path):
    image = cv2.imread(path)
    resized_img = cv2.resize(image, (640, 640))
    cv2.imwrite("resized_image.png", resized_img)
