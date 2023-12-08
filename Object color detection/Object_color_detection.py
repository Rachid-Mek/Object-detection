import os
import cv2
import numpy as np
from Fonctions import *

def object_color_detection(img, color):
    # hsv = bgr_to_hsv(img)  # Convert image to HSV color space
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define lower and upper limits of the specified color
    color_lo, color_hi = color_to_hsv_range(color)


    height, width = hsv.shape[0], hsv.shape[1]  # Get image dimensions
    mask = np.zeros((height, width), dtype=bool)  # Initialize mask
    points = []  # Initialize list to store points of detected object
    color_lo = np.array([95, 80, 50])    # Lower bound for blue in HSV
    color_hi = np.array([115, 255, 255])
    # Iterate through each pixel and apply the condition
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            pixel = hsv[i, j]  # Get pixel color
            if check_color(pixel, color_lo, color_hi):  # Check condition
                mask[i, j] = True  # Set pixel to True if condition is satisfied
                if abs(i-j) > 30:
                    points.append((i, j))  # Add point to the list


    # Change image to black where we found the color
    img[mask] = 255  # Set black color

    # The rest to white
    img[~mask] = 0  # Set white color

    return mask, img, points

def Launch_Object_color_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur de capture ")
        exit(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # flip the frame horizontally

        if not ret:
            print("Error in image read ")
            break

        # Resize the frame to a lower resolution
        resized_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)

        mask, img,  points = object_color_detection(resized_frame, [0, 0, 255])

        resized_img = cv2.resize(img, (0, 0), fx=10, fy=10)
        resized_mask = cv2.resize(np.array(mask, dtype=np.uint8), (0, 0), fx=10, fy=10)

        if len(points) > 0:
            print("points:", points[0])
            cv2.circle(frame, (points[0][0] * 10, points[0][1] * 10), 130, (0, 255, 0), 5)
            cv2.putText(frame, "x: {}, y: {}".format(points[0][0] * 10, points[0][1] * 10),
                        (10, resized_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)

        cv2.imshow('Detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------------------------------

Launch_Object_color_detection()
