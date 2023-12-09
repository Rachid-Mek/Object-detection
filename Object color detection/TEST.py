import cv2
import numpy as np 
from Cam_detection import detect_object # import the function detect_object from Cam_detection.py
from Fonctions import * # import all functions from Fonctions.py
from Object_color_detection import  object_color_detection
from Invisibility_cloak import invisibility_cloak
from Green_screen import Green_screen

# ----------------------------------------------------------------------------------------------------------------------------
frame = cv2.imread("Images/Blue_object_2.jpg" ,cv2.IMREAD_COLOR) # Read the image
image_test = frame.copy() # Copy the image
resized_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)

img, mask, points = detect_object(resized_frame) # Detect the object in the image
mask = cv2.resize(mask, (0, 0), fx=10, fy=10) # Resize the mask
background = cv2.imread("Images/Green_screen.png", cv2.IMREAD_COLOR) # Read the background image
background = cv2.resize(background, (frame.shape[1], frame.shape[0])) # Ensure the background has the correct size

frame = Green_screen(frame, points, background, mask) # Apply green screen effect to the detected object
cv2.imshow('Green screen', frame) # Show the result
cv2.imshow("mask", mask) # Show the mask

image = np.hstack((image_test, frame))
cv2.imwrite("Images/Tests/Green_screen.jpg", image) # Save the image
 
cv2.waitKey(0) # Wait for a key to be pressed
cv2.destroyAllWindows() # Destroy all windows
# ----------------------------------------------------------------------------------------------------------------------------
