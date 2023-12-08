import cv2
import numpy as np
from Cam_detection import detect_object
from Fonctions import *

 
# ----------------------------------------------------------------------------------------------------------------------------
def Green_screen(frame, points, background, mask):
    expansion_pixels=2
    x, y = int(points[0][0] * 10), int(points[0][1] * 10) # Get the coordinates of the object

    # Extract dimensions of the object from the mask
    mask = cv2.resize(mask, (0, 0), fx=10, fy=10)
    object_height, object_width = mask.shape[:2]

    # Expand the mask
    expanded_mask = expand_mask(mask, expansion_pixels)

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
    expanded_mask = cv2.resize(expanded_mask, (x_end - x_start, y_end - y_start))

    # Create a mask for the object
    mask_inv = np.zeros(expanded_mask.shape, dtype=np.uint8)
    mask_inv[expanded_mask == 0] = 255

    # Extract the region of interest from the background image
    background_region = background[y_start:y_end, x_start:x_end]

    # Blend the object region with the background region using the mask
    blended_region = np.zeros(frame_region.shape, dtype=np.uint8)
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

    return background

def expand_mask(mask, expansion_pixels=4):
    # Pad the mask with zeros
    expanded_mask = np.zeros((mask.shape[0] + 2 * expansion_pixels, mask.shape[1] + 2 * expansion_pixels), dtype=np.uint8)
    expanded_mask[expansion_pixels:expansion_pixels + mask.shape[0], expansion_pixels:expansion_pixels + mask.shape[1]] = mask
    return expanded_mask





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
