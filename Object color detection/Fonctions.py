# ----------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------

def color_to_hsv_range(color):
    """Converts a color to a range of HSV values.
    Parameters
    -----------
    - color (tuple): RGB color to convert.
    Returns
    --------
    - tuple: Lower and upper bounds of the HSV range.
    Examples
    ---------
    >>> color_to_hsv_range((0, 0, 0))
    (array([  0, 100, 100]), array([ 10, 255, 255]))
    >>> color_to_hsv_range((255, 255, 255))
    (array([  0,   0, 255]), array([ 10,  10, 255]))
     """
    # Define the RGB color
    r, g, b = color 

    # Define a threshold for the HSV range
    threshold = 10
    # Define lower and upper limits for HSV
    color_lo = np.array([r, g +100, round(b/255) +100])
    color_hi = np.array([round(b/255) +10 , b, b]) 
    return color_lo, color_hi
# ----------------------------------------------------------------------------------------------------------------------------

def get_image_dimensions(image):
    """Returns the dimensions of an image.
    Parameters
    -----------
    - image (array): Image to get the dimensions of.
    Returns
    --------
    - tuple: Height and width of the image.

    """
    # Calculate height and width of the image
    height = len(image)
    width = len(image[0]) if height > 0 else 0

    return height, width
# ----------------------------------------------------------------------------------------------------------------------------
def maximum_reduce(channels):
    """Reduces a list of arrays to a single array by taking the maximum value 
    Parameters
    -----------
    - channels (list): List of arrays to reduce.
    Returns
    --------
    - array: Maximum values of the input arrays.
    Examples
    ---------
    >>> maximum_reduce([np.array([1, 2, 3]), np.array([4, 5, 6])])
    array([4, 5, 6])
    >>> maximum_reduce([np.array([1, 2, 3]), np.array([4, 5, 2])])
    array([4, 5, 3])
    """
    result = channels[0].copy()  # Initialize the result with the first channel
    
    # Iterate through the remaining channels
    for channel in channels[1:]:
        # Compare each element and update the result with the maximum value
        for i in in_range(result.shape[0]):
            for j in in_range(result.shape[1]):
                result[i, j] = max(result[i, j], channel[i, j])
    
    return result
# ----------------------------------------------------------------------------------------------------------------------------
def minimum_reduce(channels):
    """Reduces a list of arrays to a single array by taking the minimum value 

    Parameters
    -----------
    - channels (list): List of arrays to reduce.

    Returns
    --------
    - array: Minimum values of the input arrays.

    Examples
    ---------
    >>> minimum_reduce([np.array([1, 2, 3]), np.array([4, 5, 6])])
    array([1, 2, 3])
    >>> minimum_reduce([np.array([1, 2, 3]), np.array([4, 5, 2])])
    array([1, 2, 2])
    """
    result = channels[0].copy()  # Initialize the result with the first channel
    
    # Iterate through the remaining channels
    for channel in channels[1:]:
        # Compare each element and update the result with the minimum value
        for i in in_range(result.shape[0]): 
            for j in in_range(result.shape[1]):
                result[i, j] = min(result[i, j], channel[i, j])
    
    return result
# ----------------------------------------------------------------------------------------------------------------------------
def combine_HSV(hue, saturation, value):
    """Combines hue, saturation, and value channels into a single HSV image .

    Parameters
    -----------
    - hue (array): Hue channel.
    - saturation (array): Saturation channel.
    - value (array): Value channel.

    Returns
    --------
    - array: HSV image.

    Examples
    ---------
    >>> combine_HSV(np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8]))
    array([[[0, 3, 6],
            [1, 4, 7],
            [2, 5, 8]]], dtype=uint8)
    """
    hsv_channels = np.stack((hue, saturation, value), axis=-1).astype(np.uint8)
    return hsv_channels

# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------
def expand_mask(mask, expansion_pixels=2): 
    """ Expand the mask by the number of pixels specified in expansion_pixels

    Parameters
    ----------
    - mask : mask to expand
    - expansion_pixels : number of pixels to expand the mask by

    Returns
    -------
    - expanded_mask : expanded mask


    """
    # Pad the mask with zeros
    expanded_mask = np.zeros((mask.shape[0] + 2 * expansion_pixels, mask.shape[1] + 2 * expansion_pixels), dtype=np.uint8)
    # Copy the mask to the center of the padded mask
    expanded_mask[expansion_pixels:expansion_pixels + mask.shape[0], expansion_pixels:expansion_pixels + mask.shape[1]] = mask
    return expanded_mask
# ----------------------------------------------------------------------------------------------------------------------------
def in_range(start, stop=None, step=1):
    """Returns a generator of numbers in the specified range 

    Parameters
    -----------
    - start (int): Start of the range.
    - stop (int): End of the range.
    - step (int): Step size.

    Returns
    --------
    - generator: Numbers in the specified range.

    Examples
    ---------
    >>> list(in_range(3))
    [0, 1, 2]
    >>> list(in_range(1, 4))
    [1, 2, 3]
    """
    if stop is None:
        stop = start
        start = 0
    
    while start < stop if step > 0 else start > stop:
        yield start
        start += step
# ----------------------------------------------------------------------------------------------------------------------------
def check_color(pixel, color_lo, color_hi):
    """Checks if a pixel is within a color range.
    Parameters
    -----------
    - pixel (array): Pixel value.
    - color_lo (array): Lower bound of the color range.
    - color_hi (array): Upper bound of the color range.

    Returns
    --------
    - bool: True if the pixel is within the color range, False otherwise.

    Examples
    ---------
    >>> check_color(np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([10, 10, 10]))
    True
    >>> check_color(np.array([255, 255, 255]), np.array([0, 0, 0]), np.array([10, 10, 10]))
    False
    """
    for i in in_range(len(pixel)):
        if not (pixel[i] >= color_lo[i] and pixel[i] <= color_hi[i]):
            return False
    return True
# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------
def bgr_to_hsv(img):
    """Converts an image from BGR color space to HSV color space.
    
    Parameters
    -----------
    - img (array): Image to convert.

    Returns
    --------
    - array: Image in HSV color space.

    Examples
    ---------
    >>> bgr_to_hsv(np.array([[[0, 0, 0], [255, 255, 255]]]))
    array([[[  0,   0,   0],
            [  0,   0, 255]]], dtype=uint8)
    """
    img = img / 255.0 # Convert image from [0, 255] to [0, 1]
 
    blue = img[:, :, 0] # Get blue channel
    green = img[:, :, 1] # Get green channel
    red = img[:, :, 2] # Get red channel

    # Initialize HSV channels
    hue = np.zeros_like(blue, dtype=np.float32) 
    saturation = np.zeros_like(blue, dtype=np.float32) 
    value = np.zeros_like(blue, dtype=np.float32)

    value = maximum_reduce([red, green, blue])   
    delta = value - minimum_reduce([red, green, blue])
    
    saturation = delta / (value + 1e-07) # Compute saturation channel
 
 
    for i in in_range(red.shape[0]): # Traverse rows
        for j in in_range(red.shape[1]):
            if delta[i, j] != 0:
                if value[i, j] == red[i, j]:
                    hue[i, j] = (green[i, j] - blue[i, j]) / delta[i, j]
                elif value[i, j] == green[i, j]:
                    hue[i, j] = 2.0 + (blue[i, j] - red[i, j]) / delta[i, j]
                elif value[i, j] == blue[i, j]:
                    hue[i, j] = 4.0 + (red[i, j] - green[i, j]) / delta[i, j]
            else:
                hue[i, j] = 0.0

    hue = (hue / 6.0) % 1.0  # Normalize hue to range [0, 1]

    # Scale channels to the appropriate ranges
    hue *= 179  # Scaling hue to 0-179 (OpenCV convention)
    saturation *= 255  # Scaling saturation to 0-255
    value *= 255  # Scaling value to 0-255

    # Combine HSV channels
    # hsv_img = np.stack((hue, saturation, value), axis=-1).astype(np.uint8)
    hsv_img = combine_HSV(hue, saturation, value)
    return hsv_img

# ----------------------------------------------------------------------------------------------------------------------------
def pad_image(image, pad):
    """Pads an image with a border of zeros.

    Parameters
    -----------
    - image (array): Image to pad.
    - pad (int): Size of the border.

    Returns
    --------
    - array: Padded image.

    Examples
    ---------
    >>> pad_image(np.array([[[0, 0, 0], [255, 255, 255]]]), 1)
    array([[[  0,   0,   0],
            [  0,   0,   0],
            [255, 255, 255],
            [255, 255, 255],
            [  0,   0,   0]]], dtype=uint8) 
     """
    height, width, channels = image.shape
    padded_height = height + 2 * pad
    padded_width = width + 2 * pad

    padded_image = np.zeros((padded_height, padded_width, channels), dtype=image.dtype)

    # Copy the original image to the center of the padded image
    padded_image[pad:padded_height-pad, pad:padded_width-pad] = image

    # Fill the borders with the nearest pixel values from the original image
    padded_image[:pad, pad:padded_width-pad] = image[0]  # Top border
    padded_image[padded_height-pad:, pad:padded_width-pad] = image[-1]  # Bottom border
    padded_image[:, :pad] = padded_image[:, pad:pad+1]  # Left border
    padded_image[:, padded_width-pad:] = padded_image[:, padded_width-pad-1:padded_width-pad]  # Right border

    return padded_image
# ----------------------------------------------------------------------------------------------------------------------------
 
# ----------------------------------------------------------------------------------------------------------------------------
def Apply_blur(image, kernel_size):
    """Applies a blur filter to an image.

    Parameters
    -----------
    - image (array): Image to apply the blur filter to.
    - kernel_size (int): Size of the blur kernel.

    Returns
    --------
    - array: Blurred image.

    Examples
    ---------
    >>> Apply_blur(np.array([[[0, 0, 0], [255, 255, 255]]]), 3)
    array([[[  0,   0,   0],
            [255, 255, 255]]], dtype=uint8)
    """
    height, width, channels = image.shape # Get the dimensions of the image
    image_blurred = np.copy(image) # Initialize the blurred image
    pad = kernel_size // 2  # Padding size based on the kernel size

    # Define the blur kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) # Initialize the kernel with ones
    kernel /= (kernel_size * kernel_size) # Normalize the kernel

    # padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    padded_image=pad_image(image,pad) # Pad the image with zeros , padding is an operation that adds a border around the image

    # Apply the blur filter
    for i in in_range(pad, height + pad): # Traverse rows
        for j in in_range(pad, width + pad): # Traverse columns
            for channel in in_range(channels): # Traverse channels
                # Apply the kernel filter to the defined pixel region
                image_blurred[i - pad, j - pad, channel] = np.sum(
                    padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1, channel] * kernel # Element-wise multiplication
                )

    return image_blurred.astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------------------- 
def Verify_bounds(img, lower_bound, upper_bound):
    """Returns a mask where pixels are within the specified range.
    Parameters
    -----------
    - img (array): Image to get the mask of.
    - lower_bound (array): Lower bound of the range.
    - upper_bound (array): Upper bound of the range.
    Returns
    --------
    - array: Mask where pixels are within the specified range.
    Examples
    ---------
    >>> inRange(np.array([[[0, 0, 0], [255, 255, 255]]]), np.array([0, 0, 0]), np.array([10, 10, 10]))
    array([[[  0,   0,   0],
            [255, 255, 255]]], dtype=uint8)
    """

    
    # Create a mask where pixels are within the specified range
    mask = (img >= lower_bound) & (img <= upper_bound) # Check if the pixel value lies within the specified range
    
    # Convert boolean mask to integers (0s and 255s)
    mask = mask.astype(np.uint8) * 255 # Set the pixel to white (255) in the mask
    
    return mask

# ____________________________________________________________________________________________________________________________________________________________________________________
def threshold(image, lo, hi):
    """ Returns a binary mask where pixels are within the specified range.

    Parameters
    -----------
    - image (array): Image to get the mask of.
    - lo (array): Lower bound of the range.
    - hi (array): Upper bound of the range.

    Returns
    --------
    - array: Binary mask where pixels are within the specified range.

    Examples
    ---------
    >>> threshold(np.array([[[0, 0, 0], [255, 255, 255]]]), np.array([0, 0, 0]), np.array([10, 10, 10]))
    array([[0, 0, 0], [255, 255, 255]], dtype=uint8)

    """
    height, width, _ = image.shape # Get the dimensions of the image
    binary_mask = np.zeros((height, width), dtype=np.uint8) # Initialize a binary mask

    for i in in_range(height): # Traverse rows
        for j in in_range(width): # Traverse columns
            # Check if the pixel value lies within the specified range
            if lo[0] <= image[i, j, 0] <= hi[0] and lo[1] <= image[i, j, 1] <= hi[1] and lo[2] <= image[i, j, 2] <= hi[2]:
                binary_mask[i, j] = 255  # Set the pixel to white (255) in the mask

    return binary_mask
# ____________________________________________________________________________________________________________________________________________________________________________________
def detect_contours(binary_mask):
    """Detects contours in a binary mask. 

    Parameters
    -----------
    - binary_mask (array): Binary mask to detect contours in.

    Returns
    --------
    - list: List of contours.
    - list: List of centroids of the contours.

    Examples
    ---------
    >>> detect_contours(np.array([[0, 0, 0], [255, 255, 255]]))
    ([[(1, 0), (1, 1)]], [(1, 0)])
    """
    contours = [] # Initialize a list of contours
    height, width = binary_mask.shape # Get the dimensions of the binary mask

    # Define neighbors for 8-connectivity
    neighbors = [(i, j) for i in in_range(-1, 2) for j in in_range(-1, 2) if not (i == 0 and j == 0)] # 8-connectivity

    visited = set()  # Track visited pixels to avoid repetition

    # Traverse the binary mask to detect contours using depth-first search
    for i in in_range(height): # Traverse rows
        for j in in_range(width): # Traverse columns
            if binary_mask[i, j] == 255 and (i, j) not in visited: # Check if the pixel is white and not visited
                contour = []  # Initialize a new contour
                stack = [(i, j)]  # Initialize a stack for depth-first search

                while stack: # While the stack is not empty
                    current_pixel = stack.pop() # Pop the top pixel from the stack
                    contour.append(current_pixel) # Add the pixel to the contour
                    visited.add(current_pixel) # Mark the pixel as visited

                    # Check neighbors for 8-connectivity
                    for neighbor in neighbors: # Traverse neighbors
                        x, y = current_pixel[0] + neighbor[0], current_pixel[1] + neighbor[1] # Get neighbor coordinates
                        if 0 <= x < height and 0 <= y < width and binary_mask[x, y] == 255 and (x, y) not in visited: # Check if the neighbor is white and not visited
                            stack.append((x, y))


                contours.append(contour)  # Add detected contour to the list

    # Compute centroids for each contour
    centroids = [] # Initialize a list of centroids
    for contour in contours: # Traverse contours
        centroid_x = sum(pixel[1] for pixel in contour) // len(contour) # Compute x-coordinate of the centroid
        centroid_y = sum(pixel[0] for pixel in contour) // len(contour) # Compute y-coordinate of the centroid
        # if abs(centroid_x-centroid_y)>25: # Check if the centroid is not on the diagonal
        centroids.append((centroid_x, centroid_y)) # Add the centroid to the list

    return centroids # Return the list of contours and centroids

 
# ----------------------------------------------------------------------------------------------------------------------------
def add_weighted(image1, alpha1, image2, alpha2): 
    """
    Perform weighted addition of two images: output = alpha1 * image1 + alpha2 * image2 
    
    Parameters
    -----------
    - image1 (array): First image to add.
    - alpha1 (float): Weight of the first image.
    - image2 (array): Second image to add.
    - alpha2 (float): Weight of the second image.

    Returns
    --------
    - array: Weighted addition of the two images.
    """
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same shape")

    result = np.empty_like(image1, dtype=np.float32)
    result[:, :, 0] = alpha1 * image1[:, :, 0] + alpha2 * image2[:, :, 0]  # Add the first channel
    result[:, :, 1] = alpha1 * image1[:, :, 1] + alpha2 * image2[:, :, 1]   # Add the second channel
    result[:, :, 2] = alpha1 * image1[:, :, 2] + alpha2 * image2[:, :, 2]   # Add the third channel

    result = np.clip(result, 0, 255).astype(np.uint8)  # Clip the result to the range [0, 255]
 
    return result

# ----------------------------------------------------------------------------------------------------------------------------
 
def find_contours(mask): 
    """Finds contours in a binary mask.

    Parameters
    -----------
    - mask (array): Binary mask to find contours in.

    Returns
    --------
    - list: List of contours.
    """
    # Find contours without using cv2
    contours = [] # Initialize a list of contours
    current_contour = [] # Initialize a new contour
    for i in in_range(mask.shape[0]): # Traverse rows
        for j in in_range(mask.shape[1]): # Traverse columns
            if mask[i, j] == 255: # Check if the pixel is white
                current_contour.append((i, j)) # Add the pixel to the current contour
    contours.append(np.array(current_contour)) # Add the current contour to the list
    return contours # Return the list of contours
# ----------------------------------------------------------------------------------------------------------------------------
# =============================================================================================================================
# _____________________________________________________GAME____________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------
# =============================================================================================================================
import cv2
# _____________________________________________________GAME____________________________________________________________________________________________________________________
car = cv2.imread('Object color detection/Images/car.png', cv2.IMREAD_UNCHANGED)

def move_left(car_pos_x, step=10):
    """Moves the car to the left by the specified number of pixels.

    Parameters
    -----------
    - car_pos_x (int): Current x-coordinate of the car.
    - step (int): Number of pixels to move the car by.

    Returns
    --------
    - int: New x-coordinate of the car.
    """

    car_pos_x -= step
    return max(car_pos_x, 0)

def move_right(car_pos_x, window_width, step=10):
    """Moves the car to the right by the specified number of pixels.

    Parameters
    -----------
    - car_pos_x (int): Current x-coordinate of the car.
    - window_width (int): Width of the window.
    - step (int): Number of pixels to move the car by.

    Returns
    --------
    - int: New x-coordinate of the car.

    Examples
    ---------
    >>> move_right(0, 100, 10)
    10
    >>> move_right(100, 100, 10)
    100
    """
    car_pos_x += step
    return min(car_pos_x, window_width - car.shape[1])
# ____________________________________________________________________________________________________________________________________________________________________________________

def resize_image_2d(image, scale_factor):
    """Resize a 2D image using a specified scale factor.

    Parameters
    -----------
    - image (array): 2D image to be resized.
    - scale_factor (float): Scaling factor for resizing the image.

    Returns
    --------
    - array: Resized 2D image.

    Examples
    ---------
    >>> resize_image_2d(np.array([[0, 0, 0], [255, 255, 255]]), 0.1)
    array([[  0,   0,   0],
        [255, 255, 255]], dtype=uint8)
    """
    height, width = image.shape[:2] # Get the dimensions of the image
    new_height = int(height * scale_factor) # Compute the new height
    new_width = int(width * scale_factor) # Compute the new width
    resized_image = np.zeros((new_height, new_width), dtype=np.uint8) # Initialize the resized image
    
    for i in in_range(new_height): # Traverse rows
        for j in in_range(new_width):   # Traverse columns
            resized_image[i, j] = image[int(i / scale_factor), int(j / scale_factor)] # Resize the image
    
    return resized_image    # Return the resized image

def resize_image_3d(image, scale_factor):
    """Resize a 3D image using a specified scale factor.

    Parameters
    -----------
    - image (array): 3D image to be resized.
    - scale_factor (float): Scaling factor for resizing the image.

    Returns
    --------
    - array: Resized 3D image.

    Examples
    ---------
    >>> resize_image_3d(np.array([[[0, 0, 0], [255, 255, 255]]]), 0.1)
    array([[[  0,   0,   0],
            [255, 255, 255]]], dtype=uint8)
    """
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8) # Initialize the resized image
    
    for i in in_range(new_height):
        for j in in_range(new_width):
            for k in in_range(image.shape[2]):
                resized_image[i, j, k] = image[int(i / scale_factor), int(j / scale_factor), k] # Resize the image
    
    return resized_image

def check_collision(car_pos_x, car_pos_y, car_width, car_height, obstacle_pos_x, obstacle_pos_y, obstacle_width, obstacle_height):
    """Checks if the car collides with the obstacle.

    Parameters
    -----------
    - car_pos_x (int): x-coordinate of the car.
    - car_pos_y (int): y-coordinate of the car.
    - car_width (int): Width of the car.
    - car_height (int): Height of the car.
    - obstacle_pos_x (int): x-coordinate of the obstacle.
    - obstacle_pos_y (int): y-coordinate of the obstacle.
    - obstacle_width (int): Width of the obstacle.
    - obstacle_height (int): Height of the obstacle.

    Returns
    --------
    - bool: True if the car collides with the obstacle, False otherwise.

    Examples
    ---------
    >>> check_collision(0, 0, 10, 10, 0, 0, 10, 10)
    True
    >>> check_collision(0, 0, 10, 10, 20, 20, 10, 10)
    False
    """
    '''
    Check if the car and the obstacle collide or not

    Parameters:
    -----------
    - car_pos_x (int): X-coordinate of the car.
    - car_pos_y (int): Y-coordinate of the car.
    - car_width (int): Width of the car.
    - car_height (int): Height of the car.
    - obstacle_pos_x (int): X-coordinate of the obstacle.
    - obstacle_pos_y (int): Y-coordinate of the obstacle.
    - obstacle_width (int): Width of the obstacle.
    - obstacle_height (int): Height of the obstacle.

    Returns:
    --------
    - bool: True if the car and the obstacle collide, False otherwise.

    Examples:
    ---------
    >>> check_collision(0, 0, 10, 10, 0, 0, 10, 10)
    True
    '''
    
    
    # Coordonnées de la voiture et de l'obstacle
    car_left, car_right, car_top, car_bottom = car_pos_x, car_pos_x + car_width, car_pos_y, car_pos_y + car_height
    obstacle_left, obstacle_right, obstacle_top, obstacle_bottom = (
        obstacle_pos_x, obstacle_pos_x + obstacle_width, obstacle_pos_y, obstacle_pos_y + obstacle_height
    )

    # Vérifier la collision
    return (
        car_right > obstacle_left and car_left < obstacle_right and
        car_bottom > obstacle_top and car_top < obstacle_bottom
    )  # Check if the car collides with the obstacle

 
# ----------------------------------------------------------------------------------------------------------------------------


def resize_image(image, new_size):
    """
    Resize the input image to the specified new size.

    Parameters
    - image: The input image as a NumPy array.
    - new_size: A tuple (width, height) specifying the new size of the image.

    Returns
    - The resized image as a NumPy array.
    """
    if len(image.shape) == 2:
        # Grayscale image
        return _resize_grayscale(image, new_size)
    elif len(image.shape) == 3:
        # Color image
        return _resize_color(image, new_size)
    else:
        raise ValueError("Unsupported image shape. Only 2D or 3D images are supported.")

def _resize_grayscale(image, new_size):
    """
    Resize a grayscale image to the specified new size.

    Parameters
    - image: The input grayscale image as a NumPy array.
    - new_size: A tuple (width, height) specifying the new size of the image.

    Returns
    - The resized grayscale image as a NumPy array.
    """
    new_width, new_height = new_size
    resized_image = np.zeros((new_height, new_width), dtype=image.dtype)

    for i in in_range(new_height):
        for j in in_range(new_width):
            x = j * (image.shape[1] - 1) / (new_width - 1) # Scale the x-coordinate
            y = i * (image.shape[0] - 1) / (new_height - 1) # Scale the y-coordinate

            x_low, y_low = int(np.floor(x)), int(np.floor(y)) # Get the lower bound
            x_high, y_high = int(np.ceil(x)), int(np.ceil(y)) # Get the upper bound

            x_lerp = x - x_low
            y_lerp = y - y_low

            top_left = image[y_low, x_low]
            top_right = image[y_low, x_high]
            bottom_left = image[y_high, x_low]
            bottom_right = image[y_high, x_high]

            interpolated_value = (1 - x_lerp) * (1 - y_lerp) * top_left + \
                                 x_lerp * (1 - y_lerp) * top_right + \
                                 (1 - x_lerp) * y_lerp * bottom_left + \
                                 x_lerp * y_lerp * bottom_right

            resized_image[i, j] = interpolated_value

    return resized_image
# ----------------------------------------------------------------------------------------------------------------------------
def _resize_color(image, new_size):
    """
    Resize a color image to the specified new size.

    Parameters:
    - image: The input color image as a NumPy array.
    - new_size: A tuple (width, height) specifying the new size of the image.

    Returns:
    - The resized color image as a NumPy array.
    """
    channels = image.shape[2]
    resized_channels = [_resize_grayscale(image[:, :, c], new_size) for c in in_range(channels)]
    return np.stack(resized_channels, axis=2)
# ----------------------------------------------------------------------------------------------------------------------------