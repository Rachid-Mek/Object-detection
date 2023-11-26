import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------

def color_to_hsv_range(color):
    """Converts a color to a range of HSV values."""
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
    """Returns the dimensions of an image."""
    # Calculate height and width of the image
    height = len(image)
    width = len(image[0]) if height > 0 else 0

    return height, width
# ----------------------------------------------------------------------------------------------------------------------------
def maximum_reduce(channels):
    """Reduces a list of arrays to a single array by taking the maximum value."""
    # Assuming channels is a list of arrays (e.g., red, green, blue)
    result = channels[0].copy()  # Initialize the result with the first channel
    
    # Iterate through the remaining channels
    for channel in channels[1:]:
        # Compare each element and update the result with the maximum value
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = max(result[i, j], channel[i, j])
    
    return result
# ----------------------------------------------------------------------------------------------------------------------------
def minimum_reduce(channels):
    """Reduces a list of arrays to a single array by taking the minimum value."""
    # Assuming channels is a list of arrays (e.g., red, green, blue)
    result = channels[0].copy()  # Initialize the result with the first channel
    
    # Iterate through the remaining channels
    for channel in channels[1:]:
        # Compare each element and update the result with the minimum value
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = min(result[i, j], channel[i, j])
    
    return result
# ----------------------------------------------------------------------------------------------------------------------------
def combine_HSV(hue, saturation, value):
    """Combines hue, saturation, and value channels into a single HSV image."""
    # Assuming hue, saturation, and value are arrays of the same shape
    hsv_channels = np.stack((hue, saturation, value), axis=-1).astype(np.uint8)
    return hsv_channels

# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

def in_range(start, stop=None, step=1):
    """Returns a generator of numbers in the specified range."""
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
    """Converts an image from BGR color space to HSV color space."""
    img = img / 255.0 # Convert image from [0, 255] to [0, 1]
 
    blue = img[:, :, 0] # Get blue channel
    green = img[:, :, 1] # Get green channel
    red = img[:, :, 2] # Get red channel

    # Initialize HSV channels
    hue = np.zeros_like(blue, dtype=np.float32) 
    saturation = np.zeros_like(blue, dtype=np.float32)
    value = np.zeros_like(blue, dtype=np.float32)

    value = maximum_reduce([red, green, blue])   
    # value = np.maximum.reduce([red, green, blue])
 


    # delta = value - np.minimum.reduce([red, green, blue])   
    delta = value - minimum_reduce([red, green, blue])
    # Compute delta value without numpy
    
    saturation = delta / (value + 1e-07) # Compute saturation channel
 
    # hue = np.where(delta != 0, 
    #                np.where(value == red, (green - blue) / delta,
    #                         np.where(value == green, 2.0 + (blue - red) / delta,
    #                                  4.0 + (red - green) / delta)),
    #                0.0) 
    for i in range(red.shape[0]):
        for j in range(red.shape[1]):
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