import cv2


def resize_image(path):
    """resize image to (640, 640), interpolation can be configured"""
    image = cv2.imread(path)
    resized_img = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite("resized_image.png", resized_img)
    return resized_img


def load_image_and_display(path):
    """load an image and display it"""
    img = cv2.imread(path)
    cv2.imshow("Loaded Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_image_color(path):
    """read image and convert color"""
    img = cv2.imread(path)
    converted_img = cv2.cvtColor(img, cv2.COLER_BGR2GRAY)
    return converted_img


def save_image(path):
    """read image and save it"""
    img = cv2.imread(path)
    cv2.imwrite("new_img.jpg", img)


def rotate_image(path):
    """read image and rotate it"""
    img = cv2.imread(path)
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    rotated_img = cv2.warpAffine(img, M, (cols, rows))
    return rotated_img
