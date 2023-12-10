import cv2
import numpy as np 
from Cam_detection import detect_object # import the function detect_object from Cam_detection.py
from Fonctions import * # import all functions from Fonctions.py

 
# ----------------------------------------------------------------------------------------------------------------------------
def invisibility_cloak(frame, background, points, mask):
    """ Make the object disappear from the frame and replace it with the background
    
    Parameters
    ----------
    - frame : frame to make the object disappear from it
    - background : background to replace the object with it
    - points : list of the points of the object detected
    - mask : mask of the object detected

    Returns
    -------
    - frame : frame with the object disappeared


    """
    x, y = int(points[0][0] * 10), int(points[0][1] * 10) # Get the coordinates of the object

    mask = resize_image_2d(mask, 10) # Extract dimensions of the object from the mask 

    w_frame, h_frame = frame.shape[:2] # Extract dimensions of the frame
    expand_mask(mask ,2)
    for i in in_range(w_frame): # Iterate through the frame
        for j in in_range(h_frame): # Iterate through the frame
            if mask[i, j] == 255: # Check if the pixel is part of the object
                frame[i, j] = background[i, j] # Replace the pixel with the corresponding pixel from the background
            
                # for di in in_range(-7, 8): # Iterate through the frame
                #     for dj in in_range(-7, 8): # Iterate through the frame 
                #             ni, nj = i + di, j + dj # Get the new coordinates
                #             if 0 <= ni < w_frame and 0 <= nj < h_frame and mask[ni, nj] != 255: # Check if the pixel is not part of the object
                #                 frame[ni, nj] = background[ni, nj] # Replace the pixel with the corresponding pixel from the background

    return frame # Return the frame with the object disappeared
 
# ----------------------------------------------------------------------------------------------------------------------------
# def invisibility_cloak(frame, background, points, mask):
#     x, y = int(points[0][0] * 10), int(points[0][1] * 10)

#     # Extract dimensions of the object from the mask 
#     mask = cv2.resize(mask, (0, 0), fx=10, fy=10)
#     w,h= mask.shape[:2]    
#     w_frame, h_frame = frame.shape[:2]
#     for i in range(w_frame):
#         for j in range(h_frame):
#             if mask[i,j] == 255:
#                 frame[i,j] = background[i ,j]
#     return frame 

# ----------------------------------------------------------------------------------------------------------------------------
def capture_background(): 
    """ Capture the background and return it
    Returns:
    -------
    - background : background captured
    """
    cap = cv2.VideoCapture(0) # Capture the camera
    if not cap.isOpened(): # Check if the camera is opened
        print("Erreur de capture ") # Print an error message if the camera is not opened
        exit(0) # Exit the program

    while True:
        ret, background = cap.read()
        cv2.flip(background, 1, background)

        cv2.imshow('Capture Background - Press s to start', background) # Show the background

        if cv2.waitKey(20) & 0xFF == ord('s'): # Press s to start
            break 

    cap.release() # Release the camera
    cv2.destroyAllWindows() # Close all windows
    return background # Return the background
# ----------------------------------------------------------------------------------------------------------------------------
def Launch_Invisibility_cloak():
    """ Launch the camera and detect the object in the image captured by the camera
    """
    background = capture_background() # Capture the background

    cap = cv2.VideoCapture(0) # Capture the camera
    if not cap.isOpened(): # Check if the camera is opened
        print("Erreur de capture ") # Print an error message if the camera is not opened
        exit(0) # Exit the program

    while cap.isOpened(): # Iterate through the frames captured by the camera
        ret, frame = cap.read() # read the frame from the camera
        cv2.flip(frame, 1, frame) # flip the frame horizontally , 1: flip the frame vertically , -1: flip both , 0: no flip

        # Resize the frame to a lower resolution
        resized_frame = resize_image_3d(frame, 0.1) # resize the frame to a lower resolution
     
        img, mask, points = detect_object(resized_frame) # detect the object in the frame         
        if len(points) > 0 : # if the object is detected
           frame= invisibility_cloak(frame, background, points,mask)
           cv2.imshow('Invisibility Cloak', frame) # Show the result

        if mask is not None: # if the mask is not empty
            # mask = cv2.resize(mask, (0, 0), fx=10, fy=10) # resize the mask to a lower resolution
            mask = resize_image_2d(mask, 10) # resize the mask
            cv2.imshow('mask', mask) # Show the mask

        if cv2.waitKey(20) & 0xFF == ord('q'): # Press q to exit
            break 

    cap.release() # Release the camera
    cv2.destroyAllWindows() # Close all windows

# ----------------------------------------------------------------------------------------------------------------------------
Launch_Invisibility_cloak()  # Launch the camera and detect the object in the image captured by the camera
# ----------------------------------------------------------------------------------------------------------------------------
