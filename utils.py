import cv2
import numpy as np


def dlib_shape_to_np_array(shape):
    # Init empty array
    arr = np.zeros((shape.num_parts, 2), dtype=np.int)

    # Convert each landmark to a (x, y) tuple
    for i in range(0, shape.num_parts):
        arr[i] = (shape.part(i).x, shape.part(i).y)
    return arr


def rotate_image(image, angle):
    # Get Dimensions and find center
    (height, width) = image.shape[:2]
    (centerX, centerY) = (width / 2., height / 2.)

    # Get Rotation Matrix and sine & cosine
    matrix = cv2.getRotationMatrix2D((centerX, centerY), -angle, 1.0)  # Negative angle for clockwise rotation
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])

    # Calculate new dimensions
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # Adjust rotation matrix to take dimension transformation into account
    matrix[0, 2] += (new_width / 2) - centerX
    matrix[1, 2] += (new_height / 2) - centerY

    # Rotate the image and return
    return cv2.warpAffine(image, matrix, (new_width, new_height))
