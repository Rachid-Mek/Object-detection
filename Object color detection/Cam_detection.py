import cv2 
import numpy as np
from Fonctions import *

# ----------------------------------------------------------------------------------------------------------------------------
# Global variables
# color range 
# lo = np.array([100, 50, 50])    # Lower bound for blue in HSV
# hi = np.array([130, 255, 255])   # Upper bound for blue in HSV
# yellow
# lo = np.array([20, 100, 100])    # Lower bound for blue in HSV
# hi = np.array([30, 255, 255])   # Upper bound for blue in HSV
 
lo = np.array([95, 80, 50])    # Lower bound for blue in HSV
hi = np.array([115, 255, 255])
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
    Kernel_size =5
    # img = bgr_to_hsv(img) # convert the image from BGR to HSV 
    img= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred_img =Apply_blur(img , Kernel_size) # apply blur to the image 
    binary_mask = threshold(img , lo , hi) # apply threshold to the image
    centroids = detect_contours(binary_mask) # detect the contours of the object in the image

    return blurred_img, binary_mask, centroids # return the image with the object detected , the mask and the points of the object detected
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

        # Resize the frame to a lower resolution
        resized_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)

        img, mask, points = detect_object(resized_frame)

        if(len(points) > 0):
            print("points:", points[0])
            cv2.circle(frame, (points[0][0] * 10, points[0][1] * 10), 130 , (0, 255, 0), 5)
            cv2.putText(frame, "x: {}, y: {}".format(points[0][0] * 10, points[0][1] * 10), (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)
            

        if mask is not None:
            mask = cv2.resize(mask, (0, 0), fx=10, fy=10)
            cv2.imshow('mask', mask)
        cv2.imshow('image', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------------------------------
# Launch()  # launch the camera and detect the object in the image captured by the camera

#  ----------------------------------------------------------------------------------------------------------------------------
