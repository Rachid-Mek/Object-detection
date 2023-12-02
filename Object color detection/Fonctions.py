import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------

def color_to_hsv_range(color):
    """Converts a color to a range of HSV values.
    Parameters:
    -----------
    - color (tuple): RGB color to convert.
    Returns:
    --------
    - tuple: Lower and upper bounds of the HSV range.
    Examples:
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
    Parameters:
    -----------
    - image (array): Image to get the dimensions of.
    Returns:
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
    Parameters:
    -----------
    - channels (list): List of arrays to reduce.
    Returns:
    --------
    - array: Maximum values of the input arrays.
    Examples:
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

    Parameters:
    -----------
    - channels (list): List of arrays to reduce.

    Returns:
    --------
    - array: Minimum values of the input arrays.

    Examples:
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

    Parameters:
    -----------
    - hue (array): Hue channel.
    - saturation (array): Saturation channel.
    - value (array): Value channel.

    Returns:
    --------
    - array: HSV image.

    Examples:
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

def in_range(start, stop=None, step=1):
    """Returns a generator of numbers in the specified range 

    Parameters:
    -----------
    - start (int): Start of the range.
    - stop (int): End of the range.
    - step (int): Step size.

    Returns:
    --------
    - generator: Numbers in the specified range.

    Examples:
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
    """Checks if a pixel is within a color range."""
    for i in range(len(pixel)):
        if not (pixel[i] >= color_lo[i] and pixel[i] <= color_hi[i]):
            return False
    return True
# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------
def bgr_to_hsv(img):
    # local documentation 
    """Converts an image from BGR color space to HSV color space.
    
    Parameters:
    -----------
    - img (array): Image to convert.

    Returns:
    --------
    - array: Image in HSV color space.

    Examples:
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

    Parameters:
    -----------
    - image (array): Image to pad.
    - pad (int): Size of the border.

    Returns:
    --------
    - array: Padded image.

    Examples:
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
def Apply_blur(image, kernel_size):
    """Applies a blur filter to an image.

    Parameters:
    -----------
    - image (array): Image to apply the blur filter to.
    - kernel_size (int): Size of the blur kernel.

    Returns:
    --------
    - array: Blurred image.

    Examples:
    ---------
    >>> Apply_blur(np.array([[[0, 0, 0], [255, 255, 255]]]), 3)
    array([[[  0,   0,   0],
            [255, 255, 255]]], dtype=uint8)
    """
    height, width, channels = image.shape
    image_blurred = np.copy(image)
    pad = kernel_size // 2  # Padding size based on the kernel size

    # Define the blur kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= (kernel_size * kernel_size)

    # Pad the image to handle border pixels
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

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
    Parameters:
    -----------
    - img (array): Image to get the mask of.
    - lower_bound (array): Lower bound of the range.
    - upper_bound (array): Upper bound of the range.
    Returns:
    --------
    - array: Mask where pixels are within the specified range.
    Examples:
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

    Parameters:
    -----------
    - image (array): Image to get the mask of.
    - lo (array): Lower bound of the range.
    - hi (array): Upper bound of the range.

    Returns:
    --------
    - array: Binary mask where pixels are within the specified range.

    Examples:
    ---------
    >>> threshold(np.array([[[0, 0, 0], [255, 255, 255]]]), np.array([0, 0, 0]), np.array([10, 10, 10]))
    array([[0, 0, 0], [255, 255, 255]], dtype=uint8)

    """
    height, width, channels = image.shape # Get the dimensions of the image
    binary_mask = np.zeros((height, width), dtype=np.uint8) # Initialize a binary mask

    for i in in_range(height):
        for j in in_range(width):
            # Check if the pixel value lies within the specified range
            if lo[0] <= image[i, j, 0] <= hi[0] and lo[1] <= image[i, j, 1] <= hi[1] and lo[2] <= image[i, j, 2] <= hi[2]:
                binary_mask[i, j] = 255  # Set the pixel to white (255) in the mask

    return binary_mask
# ____________________________________________________________________________________________________________________________________________________________________________________
def detect_contours(binary_mask):
    """Detects contours in a binary mask. 

    Parameters:
    -----------
    - binary_mask (array): Binary mask to detect contours in.

    Returns:
    --------
    - list: List of contours.
    - list: List of centroids of the contours.

    Examples:
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
        centroids.append((centroid_x, centroid_y)) # Add the centroid to the list

    return centroids

# ____________________________________________________________________________________________________________________________________________________________________________________

def resize_image_2d(image, scale_factor):
    """Resize a 2D image using a specified scale factor.

    Parameters:
    -----------
    - image (array): 2D image to be resized.
    - scale_factor (float): Scaling factor for resizing the image.

    Returns:
    --------
    - array: Resized 2D image.

    Examples:
    ---------
    >>> resize_image_2d(np.array([[0, 0, 0], [255, 255, 255]]), 0.1)
    array([[  0,   0,   0],
        [255, 255, 255]], dtype=uint8)
    """
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    resized_image = np.zeros((new_height, new_width), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            resized_image[i, j] = image[int(i / scale_factor), int(j / scale_factor)]
    
    return resized_image

def resize_image_3d(image, scale_factor):
    """Resize a 3D image using a specified scale factor.

    Parameters:
    -----------
    - image (array): 3D image to be resized.
    - scale_factor (float): Scaling factor for resizing the image.

    Returns:
    --------
    - array: Resized 3D image.

    Examples:
    ---------
    >>> resize_image_3d(np.array([[[0, 0, 0], [255, 255, 255]]]), 0.1)
    array([[[  0,   0,   0],
            [255, 255, 255]]], dtype=uint8)
    """
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            for k in range(image.shape[2]):
                resized_image[i, j, k] = image[int(i / scale_factor), int(j / scale_factor), k]
    
    return resized_image
