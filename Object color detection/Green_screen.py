import cv2
import numpy as np
from Cam_detection import detect_object
from Fonctions import * # import all functions from Fonctions.py

 
# ----------------------------------------------------------------------------------------------------------------------------
def Green_screen(frame, points, background, mask):
    """ Apply green screen effect to the detected object

    Parameters
    ----------
    - frame : frame to apply the green screen effect to it
    - points : list of the points of the object detected
    - background : background to replace the object with it
    - mask : mask of the object detected
    
    Returns
    -------
    - background : background with the green screen effect applied to it
    """
    expansion_pixels = 2  # Number of pixels to expand the mask by
    x, y = int(points[0][0] * 10), int(points[0][1] * 10)  # Get the coordinates of the object

    # Extract dimensions of the object from the mask
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    object_height, object_width = mask.shape[:2]

    # Expand the mask by the number of pixels specified in expansion_pixels
    if np.random.randint(0, 2) == 1:
        mask = expand_mask(mask, expansion_pixels)

    # Ensure the indices are within the valid range
    y_start = max(0, y - object_height - expansion_pixels)
    y_end = min(frame.shape[0], y + object_height + expansion_pixels)
    x_start = max(0, x - object_width - expansion_pixels)
    x_end = min(frame.shape[1], x + object_width + expansion_pixels)

    # Get the exact region of interest in the frame
    frame_region = frame[y_start:y_end, x_start:x_end]

    # Ensure the background has the correct size
    background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    # Resize the expanded mask to match the size of the frame region
    mask = cv2.resize(mask, (x_end - x_start, y_end - y_start))


    for i in in_range(frame_region.shape[0]):
        for j in in_range(frame_region.shape[1]):
            if mask[i, j] != 0 :
                background[y_start + i, x_start + j] = frame_region[i, j]

    return background  # Return the background with the green screen effect applied to it


# ----------------------------------------------------------------------------------------------------------------------------


def Launch_Green_screen():
    """ Launch the camera and detect the object in the image captured by the camera
    """
    cap = cv2.VideoCapture(0) # Launch the camera
    if not cap.isOpened(): # Check if the camera is opened
        print("Erreur de capture ") # Print an error message
        exit(0) # Exit the program

    # Load the green screen background image
    background = cv2.imread('Object color detection/Images/Green_screen.png')  # Replace with the path to your image

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        resized_frame = resize_image_3d(frame, 0.1) # resize the frame to a lower resolution to speed up the processing
        img, mask, points = detect_object(resized_frame) # detect the object in the frame

        if len(points) > 0: # if the object is detected 
            Green_screen_frame = Green_screen(frame, points, background, mask) # Apply green screen effect to the detected object
 
            # Display the result
            cv2.imshow(' Green Screen ', Green_screen_frame) # Show the result

        if mask is not None:  # Show the mask of the object detected
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]) )
            cv2.imshow('Mask', mask) # Show the mask of the object detected
        # cv2.imshow('Original Image', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release() # Release the camera
    cv2.destroyAllWindows() # Close all windows

# ----------------------------------------------------------------------------------------------------------------------------
# Launch_Green_screen()  # Launch the camera and detect the object in the image captured by the camera
# ----------------------------------------------------------------------------------------------------------------------------