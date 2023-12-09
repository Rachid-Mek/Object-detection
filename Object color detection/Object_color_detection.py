import cv2
import numpy as np
from Fonctions import *
# ----------------------------------------------------------------------------------------------------------------------------

def object_color_detection(img, color_lo=np.array([95, 80, 50]), color_hi=np.array([115, 255, 255])):
    """ Detect the object in the image base on its color and return the image with the object detected and the mask
    
    Parameters:
    ----------
    - img : image to detect the object in it

    Returns:
    -------
    - img : image with the object detected
    - mask : mask of the object detected
    - points : list of the points of the object detected
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert the image from BGR to HSV

    height, width = hsv.shape[0], hsv.shape[1] # Get the height and width of the image
    mask = np.zeros((height, width), dtype=np.uint8)  # Change the data type to uint8
    points = [] # List of the points of the object detected

    for i in in_range(height): # Iterate through the image
        for j in in_range(width): # Iterate through the image
            pixel = hsv[i, j] # Get the pixel value
            if check_color(pixel, color_lo, color_hi): # Check if the pixel is in the range of the color
                mask[i, j] = 255  # set to white
                if abs(i - j) > 30:  
                    points.append((i, j)) # Add the point to the list

    img[mask == 255] = 255 # Set the pixels to white
    img[mask == 0] = 0 # Set the pixels to black
 
    return mask, img, points # Return the mask, the image with the object detected and the points of the object detected

def launch_object_color_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur de capture ")
        exit(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Error in image read ")
            break
        org_frame = frame.copy()
        mask, img, points = object_color_detection(frame)

        if len(points) > 0:
            print("points:", points[0]) 
            cv2.circle(org_frame, (points[0][1], points[0][0]), 120, (0, 255, 0), 5)  # Corrected the order of points
            cv2.putText(org_frame, "x: {}, y: {}".format(points[0][1], points[0][0]),
                        (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)

        cv2.imshow('Detection result', org_frame) # Show the result

        if mask is not None:
            cv2.imshow('mask', mask) # Show the mask

        if cv2.waitKey(20) & 0xFF == ord('q'): # Press q to exit
            break 

    cap.release() # Release the camera
    cv2.destroyAllWindows() # Close all windows

# ----------------------------------------------------------------------------------------------------------------------------
# launch_object_color_detection() # Launch the camera and detect the object in the image captured by the camera
# ----------------------------------------------------------------------------------------------------------------------------
