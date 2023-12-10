import cv2 
import numpy as np
from Fonctions import *

# ----------------------------------------------------------------------------------------------------------------------------
# Global variables
# lo = np.array([20, 100, 100])    # Lower bound for yellow in HSV
# hi = np.array([30, 255, 255])    # Upper bound for yellow in HSV
 
lo = np.array([95, 80, 50])    # Lower bound for blue in HSV
hi = np.array([115, 255, 255]) # Upper bound for blue in HSV
# ----------------------------------------------------------------------------------------------------------------------------
# Functions
def detect_object(img): 
    """ Detect the object in the image and return the image with the object detected and the mask 
    
    Parameters
    ----------
    - img : image to detect the object in it

    Returns
    -------
    - img : image with the object detected
    - mask : mask of the object detected
    - points : list of the points of the object detected

    """
    Kernel_size = 5 # kernel size for the blur which is the size of the filter used for the convolution
    # img = bgr_to_hsv(img) # convert the image from BGR to HSV 
    img= cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert the image from BGR to HSV
    Apply_blur(img , Kernel_size) # apply blur to the image 
    binary_mask = threshold(img , lo , hi) # apply threshold to the image
    centroids = detect_contours(binary_mask) # detect the contours of the object in the image

    return  img ,binary_mask, centroids # return the image with the object detected , the mask and the points of the object detected
# ----------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------- 
def Launch():
    """ Launch the camera and detect the object in the image captured by the camera
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur de capture ")
        exit(0)

    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)  # flip the frame horizontally , 1: flip the frame vertically , -1: flip both , 0: no flip 
        if not ret:
            print("Error in image read ")
            break 

        # Resize the frame to a lower resolution for faster processing
        # resized_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1) 
        resized_frame = resize_image_3d(frame, 0.1) # resize the frame to a lower resolution for faster processing

        img ,mask, points = detect_object(resized_frame) # detect the object in the image captured by the camera

        if(len(points) > 0):
            print("points:", points[0]) # print the points of the object detected
            cv2.circle(frame, (points[0][0] * 10, points[0][1] * 10), 130 , (0, 255, 0), 5)
            cv2.putText(frame, "x: {}, y: {}".format(points[0][0] * 10, points[0][1] * 10), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)
            
        if mask is not None:  # if the mask is not empty
            # mask = cv2.resize(mask, (0, 0), fx=10, fy=10) # resize the mask 
            mask = resize_image_2d(mask, 10)
            cv2.imshow('mask', mask) # show the mask
        cv2.imshow('Detection', frame) # show the frame with the object detected
        

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release() # release the camera
    cv2.destroyAllWindows() # destroy all windows

# ----------------------------------------------------------------------------------------------------------------------------
# Launch()  # launch the camera and detect the object in the image captured by the camera

#  ----------------------------------------------------------------------------------------------------------------------------
