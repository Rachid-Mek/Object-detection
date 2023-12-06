import cv2 
import numpy as np
from Fonctions import *

# ----------------------------------------------------------------------------------------------------------------------------
# Global variables
# color range 
lo = np.array([100, 50, 50])    # Lower bound for blue in HSV
hi = np.array([130, 255, 255])   # Upper bound for blue in HSV

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
    img = bgr_to_hsv(img) # convert the image from BGR to HSV 
    blurred_img =Apply_blur(img , Kernel_size) # apply blur to the image 
    binary_mask = threshold(img , lo , hi) # apply threshold to the image
    centroids = detect_contours(binary_mask) # detect the contours of the object in the image

    return blurred_img, binary_mask, centroids # return the image with the object detected , the mask and the points of the object detected
# ----------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------- 
def Launch() :
    """ Launch the camera and detect the object in the image captured by the camera
    """
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur de capture ")
        exit(0)

    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.flip(frame,1,frame) # flip the frame horizontally , 1 : flip the frame vertically , -1 : flip both , 0 : no flip 
        if not ret :
            print("Error in image read ")
            break 
        img , mask , points = detect_object(frame)
        if(len(points) > 0):
            print("points :", points[0])
            cv2.circle(img,(points[0][0],points[0][1]),150,(0,255,0),5)
            cv2.putText(img, "x: {}, y: {}".format(points[0][0],points[0][1]), (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)
            cv2.imshow("img" , img)
            
        if not mask is None:
            cv2.imshow("mask" , mask)
                
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    # cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------------------------------
Launch()  # launch the camera and detect the object in the image captured by the camera

#  ----------------------------------------------------------------------------------------------------------------------------