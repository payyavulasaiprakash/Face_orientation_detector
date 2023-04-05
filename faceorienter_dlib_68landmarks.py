#importing the required dependencies
import os
import cv2,glob,csv
import dlib
import numpy as np
import utils

#loading the dlib models
FACE_DETECTOR = dlib.get_frontal_face_detector()
LANDMARK_DETECTOR = dlib.shape_predictor(
    os.path.join(os.path.dirname(__file__), 'models', 'shape_predictor_68_face_landmarks.dat')
)

def prediction(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n_rotations = 0
    face_rects = FACE_DETECTOR(img_gray, 1)
    # If no faces are detected, rotate the image by 90 degrees and try again
    if len(face_rects) == 0:
        for n_rotations in range(1, 4):
            img_gray = utils.rotate_image(img_gray, 90)
            face_rects = FACE_DETECTOR(img_gray)
            if len(face_rects) > 0:
                break
            # Find landmarks
    landmarks = None
    if len(face_rects) > 0:
        landmarks = utils.dlib_shape_to_np_array(LANDMARK_DETECTOR(img_gray, face_rects[0]))

    if landmarks is None:  # For now, just return a random orientation if no landmarks were found (not good)
        return "couldn't detect faces in the images"
    eye_l_center=((landmarks[36][0] + landmarks[39][0])/2,(landmarks[36][1] + landmarks[39][1])/2)
    eye_r_center=((landmarks[42][0] + landmarks[45][0])/2,(landmarks[42][1] + landmarks[45][1])/2)
    nose_point=(landmarks[34][0], landmarks[34][1])


    # Predict orientation
    if eye_l_center[0] >= nose_point[0] >= eye_r_center[0]:
        if nose_point[1] >= (eye_l_center[1] + eye_r_center[1]) / 2:
            orientation = 0
        else:
            orientation = 2
    else:
        if nose_point[0] >= (eye_l_center[0] + eye_r_center[0]) / 2:
            orientation = 3
        else:
            orientation = 1

    predicted_orientation = ['left', 'down', 'right','up'][(n_rotations + orientation) % 4]
    return predicted_orientation

#path of the folder which consists of images and for which we need to find the face orientation 
main_folder_path='sample_images'
images=glob.glob(main_folder_path+"/*")

with open(main_folder_path+"_predicted_orientation_68lm_dlib.csv",mode="w") as output_file:
    writer=csv.writer(output_file)
    writer.writerow(['image_path',"predicted_orientation"])
    for image_path in images:
        if os.path.isfile(image_path): #check's given path is file or not
            print(image_path)
            predicted_orientation=prediction(image_path)
            writer.writerow([os.path.basename(image_path),predicted_orientation])

    

