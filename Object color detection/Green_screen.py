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
    expansion_pixels=2 # Number of pixels to expand the mask by
    x, y = int(points[0][0] * 10), int(points[0][1] * 10) # Get the coordinates of the object

    # Extract dimensions of the object from the mask
    mask = cv2.resize(mask, (0, 0), fx=10, fy=10) # Extract dimensions of the object from the Mask
    # mask = resize_image(mask, 10,10) # Extract dimensions of the object from the Mask
    object_height, object_width = mask.shape[:2] # Extract dimensions of the object from the mask

    # Expand the mask
    if np.random.randint(0, 2) == 4: # Randomly choose whether to expand the mask or not to make the effect more realistic
        expanded_mask = expand_mask(mask, expansion_pixels) # Expand the mask by the number of pixels specified in expansion_pixels
    else:
        expanded_mask = mask
    # Ensure the indices are within the valid range
    y_start = max(0, y - object_height - expansion_pixels) # Ensure the indices are within the valid range
    y_end = min(frame.shape[0], y + object_height + expansion_pixels) # Ensure the indices are within the valid range
    x_start = max(0, x - object_width - expansion_pixels) # Ensure the indices are within the valid range
    x_end = min(frame.shape[1], x + object_width + expansion_pixels) # Ensure the indices are within the valid range
 
    # Get the exact region of interest in the frame
    frame_region = frame[y_start:y_end, x_start:x_end] # Get the exact region of interest in the frame

    # Ensure the background has the correct size
    background = cv2.resize(background, (frame.shape[1], frame.shape[0])) # Ensure the background has the correct size

    # Resize the expanded mask to match the size of the frame region
    expanded_mask = cv2.resize(expanded_mask, (x_end - x_start, y_end - y_start)) # Resize the expanded mask to match the size of the frame region

    # Create a mask for the object
    mask_inv = np.zeros(expanded_mask.shape, dtype=np.uint8) # Create a mask for the object
    mask_inv[expanded_mask == 0] = 255 # Create a mask for the object

    background_region = background[y_start:y_end, x_start:x_end] # Extract the region of interest from the background image

    # Blend the object region with the background region using the mask
    blended_region = np.zeros(frame_region.shape, dtype=np.uint8) # Blend the object region with the background region using the mask
    for i in in_range(frame_region.shape[0]): 
        for j in in_range(frame_region.shape[1]):
            if expanded_mask[i, j] != 0:
                blended_region[i, j] = frame_region[i, j]

    # Blend the background region with the inverted mask
    blended_background = np.zeros(background_region.shape, dtype=np.uint8)
    for i in in_range(background_region.shape[0]):
        for j in in_range(background_region.shape[1]):
            if mask_inv[i, j] != 0:
                blended_background[i, j] = background_region[i, j]

    result_region = blended_region + blended_background # Add the two blended images

    background[y_start:y_end, x_start:x_end] = result_region # Update the background image

    return background # Return the background with the green screen effect applied to it

 
# ----------------------------------------------------------------------------------------------------------------------------


def Launch_Green_screen():
    """ Launch the camera and detect the object in the image captured by the camera
    """
    cap = cv2.VideoCapture(0) # Launch the camera
    if not cap.isOpened(): # Check if the camera is opened
        print("Erreur de capture ") # Print an error message
        exit(0) # Exit the program

    # Load the green screen background image
    background = cv2.imread('Images/Green_screen.png')  # Replace with the path to your image

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        resized_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1) # resize the frame to a lower resolution to speed up the processing

        img, mask, points = detect_object(resized_frame) # detect the object in the frame

        if len(points) > 0: # if the object is detected 
            Green_screen_frame = Green_screen(frame,  points, background, mask) # Apply green screen effect to the detected object
 
            # Display the result
            cv2.imshow(' Green Screen ', Green_screen_frame) # Show the result

        if mask is not None:  # Show the mask of the object detected
            mask= cv2.resize(mask, (frame.shape[1], frame.shape[0])) # Resize the mask to match the size of the frame
            cv2.imshow('Mask', mask) # Show the mask of the object detected
        # cv2.imshow('Original Image', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release() # Release the camera
    cv2.destroyAllWindows() # Close all windows

# ----------------------------------------------------------------------------------------------------------------------------
# Launch_Green_screen()  # Launch the camera and detect the object in the image captured by the camera
# ----------------------------------------------------------------------------------------------------------------------------
