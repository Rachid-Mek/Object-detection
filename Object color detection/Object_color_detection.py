import os
import cv2
import numpy as np
from Fonctions import *

    
def object_color_detection(img , color) :
    """Detects objects of a specific color in an image , and changes their color to black , and the rest to white.
    this function uses the bgr_to_hsv function and the check_color function.
      you need to provide the image and the color you want to detect."""
    
    if img is None: # Check if image is loaded
        print('Could not open or find the image')
        exit(0)
    hsv = bgr_to_hsv(img)  # Convert image to HSV color space

    # Define lower and upper limits of what we call "red"
    color_lo, color_hi = color_to_hsv_range(color) # Get HSV range for red color
    # color_lo = np.array([0, 100, 100])
    # color_hi = np.array([10, 255, 255])
    height , width  = hsv.shape[0] , hsv.shape[1] # Get image dimensions
    mask = np.zeros((height, width ), dtype=bool) # Initialize mask

    # Iterate through each pixel and apply the condition
    for i in in_range(hsv.shape[0]): # Iterate through each pixel
        for j in in_range(hsv.shape[1]): # Iterate through each pixel
            pixel = hsv[i, j] # Get pixel color
            if check_color(pixel , color_lo , color_hi): # Check condition 
                mask[i, j] = True # Set pixel to True if condition is satisfied

    # Change image to black where we found the color
    img[mask] = 0  # Set black color

    # The rest to white
    img[~mask] = 255 # Set white color 
    return img

# ----------------------------------------------------------------------------------------------------------------------------
# Load image
img = cv2.imread('Object color detection/Images/red-sedan-car.jpg' , cv2.IMREAD_COLOR)
color = [0 , 0 , 255] # red color
img = object_color_detection(img ,color)
# Show image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ----------------------------------------------------------------------------------------------------------------------------