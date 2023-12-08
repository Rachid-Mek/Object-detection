import cv2
import numpy as np
from Object_color_detection import object_color_detection
from Cam_detection import detect_object
from Fonctions import *



# ----------------------------------------------------------------------------------------------------------------------------

def Green_screen(frame, points, background, mask):
    """ Apply green screen effect to the detected object
    Parameters:
    ----------
    - frame : image to apply the green screen effect to it
    - points : list of the points of the object detected
    - background : background image to replace the object with it
    - mask : mask of the object detected
    Returns:
    -------
    - background : image with the green screen effect applied to it
    """
    x, y = int(points[0][0] * 10), int(points[0][1] * 10) # Extract the coordinates of the object from the list of points

    # Extract dimensions of the object from the mask 
    mask = cv2.resize(mask, (0, 0), fx=10, fy=10) # Resize the mask to match the size of the frame region
    object_height, object_width = mask.shape[:2] # Extract dimensions of the object from the mask

    # Ensure the indices are within the valid range
    y_start = max(0, y - object_height) # Ensure the indices are within the valid range
    y_end = min(frame.shape[0], y + object_height) # Ensure the indices are within the valid range
    x_start = max(0, x - object_width) # Ensure the indices are within the valid range
    x_end = min(frame.shape[1], x + object_width) # Ensure the indices are within the valid range

    # Get the exact region of interest in the frame
    frame_region = frame[y_start:y_end, x_start:x_end] # Get the exact region of interest in the frame

    # Ensure the background has the correct size
    background = cv2.resize(background, (frame.shape[1], frame.shape[0])) # Ensure the background has the correct size

    # Resize the mask to match the size of the frame region
    mask = cv2.resize(mask, (x_end - x_start, y_end - y_start)) # Resize the mask to match the size of the frame region

    # Create a mask for the object
    # mask_inv = cv2.bitwise_not(mask)
    mask_inv = np.zeros(mask.shape, dtype=np.uint8) # Create a mask for the object
    mask_inv[mask == 0] = 255 # Create a mask for the object

    # Extract the region of interest from the background image
    background_region = background[y_start:y_end, x_start:x_end] # Extract the region of interest from the background image

    # Blend the object region with the background region using the mask
    blended_region = np.zeros(frame_region.shape, dtype=np.uint8) # Blend the object region with the background region using the mask
    for i in in_range(frame_region.shape[0]): # Blend the object region with the background region using the mask
        for j in in_range(frame_region.shape[1]): # Blend the object region with the background region using the mask
            if mask[i, j] != 0: # Blend the object region with the background region using the mask
                blended_region[i, j] = frame_region[i, j] # Blend the object region with the background region using the mask

    # blended_background = cv2.bitwise_and(background_region, background_region, mask=mask_inv)
    blended_background = np.zeros(background_region.shape, dtype=np.uint8) 
    for i in in_range(background_region.shape[0]):
        for j in in_range(background_region.shape[1]):
            if mask_inv[i, j] != 0:
                blended_background[i, j] = background_region[i, j] # Blend the object region with the background region using the mask
    # result_region = cv2.add(blended_region, blended_background)
    result_region = np.zeros(background_region.shape, dtype=np.uint8)
    for i in in_range(background_region.shape[0]):
        for j in in_range(background_region.shape[1]):
            result_region[i, j] = blended_region[i, j] + blended_background[i, j] # sum the two images

    # Replace the region in the background image with the blended result
    background[y_start:y_end, x_start:x_end] = result_region # Replace the region in the background image with the blended result

    return background




# ----------------------------------------------------------------------------------------------------------------------------


def Launch_Green_screen():
    """ Launch the camera and detect the object in the image captured by the camera
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur de capture ")
        exit(0)

    # Load the green screen background image
    background = cv2.imread('Images/Green_screen.png')  # Replace with the path to your image

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Resize the frame to a lower resolution
        resized_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)

        img, mask, points = detect_object(resized_frame)

        if len(points) > 0:
            # Apply green screen effect to the detected object
            Green_screen_frame = Green_screen(frame,  points, background, mask)

            # Resize the green screen region to the original frame size
            # Green_screen_frame = cv2.resize(Green_screen_frame,(0, 0), fx=10, fy=10)

            # Display the result
            cv2.imshow(' Green Screen ', Green_screen_frame)

        if mask is not None:
            mask= cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            cv2.imshow('Mask', mask)
        # cv2.imshow('Original Image', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------------------------------
Launch_Green_screen()  # Launch the camera and detect the object in the image captured by the camera
# ----------------------------------------------------------------------------------------------------------------------------
