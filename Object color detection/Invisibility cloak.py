import cv2
import numpy as np
from Cam_detection import detect_object
from Fonctions import *

 
# ----------------------------------------------------------------------------------------------------------------------------
def invisibility_cloak(frame, background, points, mask):
    x, y = int(points[0][0] * 10), int(points[0][1] * 10)

    # Extract dimensions of the object from the mask
    mask = cv2.resize(mask, (0, 0), fx=10, fy=10)
    w, h = mask.shape[:2]
    w_frame, h_frame = frame.shape[:2]

    # Iterate through the mask and add 2 pixels around the detected object
    for i in in_range(w_frame):
        for j in in_range(h_frame):
            if mask[i, j] == 255:
                # Set the central pixel to the background value
                frame[i, j] = background[i, j]

                # Update pixels around the central pixel (add 4 pixels)
                for di in in_range(-4, 5):
                    for dj in in_range(-4, 5):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < w_frame and 0 <= nj < h_frame and mask[ni, nj] != 255:
                            frame[ni, nj] = background[ni, nj]

    return frame
 
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur de capture ")
        exit(0)

    while True:
        ret, background = cap.read()
        cv2.flip(background, 1, background)

        cv2.imshow('Capture Background - Press s to start', background)

        if cv2.waitKey(20) & 0xFF == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return background
# ----------------------------------------------------------------------------------------------------------------------------
def Launch_Invisibility_cloak():
    background = capture_background()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur de capture ")
        exit(0)

    while cap.isOpened():
        ret, frame = cap.read() # read the frame from the camera
        cv2.flip(frame, 1, frame) # flip the frame horizontally , 1: flip the frame vertically , -1: flip both , 0: no flip

        # Resize the frame to a lower resolution
        resized_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1) # resize the frame to a lower resolution
        resized_background = cv2.resize(background, (0, 0), fx=0.1, fy=0.1) # resize the background to a lower resolution
     
        img, mask, points = detect_object(resized_frame) # detect the object in the frame         
        if len(points) > 0 : # if the object is detected
           frame= invisibility_cloak(frame, background, points,mask)
        #    frame= cv2.resize(frame, (0, 0), fx=10, fy=10)
           cv2.imshow('Invisibility Cloak', frame)

        if mask is not None:
            mask = cv2.resize(mask, (0, 0), fx=10, fy=10)
            cv2.imshow('mask', mask)
        # cv2.imshow('image', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------------------------------
Launch_Invisibility_cloak()  # Launch the camera and detect the object in the image captured by the camera
# ----------------------------------------------------------------------------------------------------------------------------
