import cv2
import numpy as np

#--TrackBar functions------------------------------------------------------------------------------------------------------------------------------------------------------

# for all filters
def changeTh(value):
    global th
    th = value
    filter()
    
# for all filters    
def changeType(x):
    global type
    type = x
    filter()
    
def changeFilter(value):
    global method
    method = label_values[value]
    filter()
    
    
 #for dilate erode morphex
 # struct rect and cross
def changeStructure(value):
    global structure
    structure = struct_values[value]
    filter()
  
# for morphex  
# open and close
def changeMorphType(value):
    global morph_type
    morph_type = morph_values[value]
    filter()

# for erode   
def change_erode_size(x):
    global sizeErode
    sizeErode = x 
    filter()

# for dilate
def change_dilate_size(x):
    global sizeDilate
    sizeDilate = x 
    filter()
# for morphex
def change_morphEx_size(x):
    global sizeMorphEx
    sizeMorphEx = x 
    filter()
#--Filter Functions TP5------------------------------------------------------------------------------------------------------------------------------------------------------

def thresholding():
    # Initialize the loop variable for the row index
    i = 0
    # Iterate over the rows of the image
    while i < height:
        # Initialize the loop variable for the column index
        j = 0
        # Iterate over the columns of the image
        while j < width:
            # Check the thresholding type and apply the corresponding method
            if type == 0:  # THRESH_BINARY
                imgRes_l[i][j] = 255 if imgRes_l[i][j] > th else 0
            elif type == 1:  # THRESH_BINARY_INV
                imgRes_l[i][j] = 0 if imgRes_l[i][j] > th else 255
            elif type == 2:  # THRESH_TRUNC
                imgRes_l[i][j] = th if imgRes_l[i][j] > th else imgRes_l[i][j]
            elif type == 3:  # THRESH_TOZERO
                imgRes_l[i][j] = imgRes_l[i][j] if imgRes_l[i][j] > th else 0
            elif type == 4:  # THRESH_TOZERO_INV
                imgRes_l[i][j] = 0 if imgRes_l[i][j] > th else imgRes_l[i][j]
            
            # Move to the next column
            j += 1
        # Move to the next row
        i += 1

        
def filter2D(kernel):
    # Initialize the loop variable for the row index
    i = 0
    # Iterate over the rows of the image
    while i < height:
        # Initialize the loop variable for the column index
        j = 0
        # Iterate over the columns of the image
        while j < width:
            # Handle boundary conditions by replicating edge pixels
            i_index = min(max(i, 1), height - 2)
            j_index = min(max(j, 1), width - 2)

            # Perform 2D convolution using the provided kernel
            pixel_value = (
                img[i_index][j_index] * kernel[1][1] +  # Center
                img[i_index - 1][j_index] * kernel[0][1] +  # Left
                img[i_index + 1][j_index] * kernel[2][1] +  # Right
                img[i_index][j_index - 1] * kernel[1][0] +  # Up
                img[i_index][j_index + 1] * kernel[1][2] +  # Down
                img[i_index - 1][j_index - 1] * kernel[0][0] +  # Upper Left
                img[i_index - 1][j_index + 1] * kernel[0][2] +  # Lower Left
                img[i_index + 1][j_index - 1] * kernel[2][0] +  # Upper Right
                img[i_index + 1][j_index + 1] * kernel[2][2]    # Lower Right
            )
            # Update the result image with the computed pixel value
            imgRes_l[i][j] = pixel_value
            # Move to the next column
            j += 1
        # Move to the next row
        i += 1

            
#--Filter Functions TP9------------------------------------------------------------------------------------------------------------------------------------------------------
#~~~Erode~~~~~~~~~~~~~~~~~~~~
""" Erosion: effectue un « et » logique entre les voisins d’un pixel (diminue le contour de l’ordre
d’un pixel)"""

def erode_cross(img, sizeErode):
    

    # Initialize the loop variables
    i = 0
    # Iterate over the rows of the image
    while i < height:
        # Initialize the inner loop variable
        j = 0
        # Iterate over the columns of the image
        while j < width:
            # Handle boundary conditions by replicating edge pixels
            i_index = min(max(i, sizeErode), height - sizeErode - 1)
            j_index = min(max(j, sizeErode), width - sizeErode - 1)

            # Perform logical AND operation between neighboring pixels based on the kernel
            pixel_values = []
            m = -sizeErode
            # Iterate over the elements of the horizontal and vertical lines in the kernel
            while m <= sizeErode:
                # Collect pixel values along the horizontal line
                pixel_values.append(img[i_index + m, j_index])
                # Collect pixel values along the vertical line
                pixel_values.append(img[i_index, j_index + m])
                m += 1

            # Find the minimum pixel value among the collected values
            pixel_value = np.min(np.array(pixel_values))
            # Update the result image with the minimum pixel value
            imgRes_l[i][j] = pixel_value
            # Move to the next column
            j += 1
        # Move to the next row
        i += 1
    return imgRes_l

        
def erode_rect(img, sizeErode):
    # Initialize the loop variable for the row index
    i = 0
    # Iterate over the rows of the image
    while i < height:
        # Initialize the loop variable for the column index
        j = 0
        # Iterate over the columns of the image
        while j < width:
            # Handle boundary conditions by replicating edge pixels
            i_index = min(max(i, sizeErode), height - sizeErode - 1)
            j_index = min(max(j, sizeErode), width - sizeErode - 1)

            # Perform logical AND operation between neighboring pixels based on the kernel
            pixel_values = []
            m = -sizeErode
            # Iterate over the elements of the square kernel
            while m <= sizeErode:
                n = -sizeErode
                while n <= sizeErode:
                    # Collect pixel values within the square region of the kernel
                    pixel_values.append(img[i_index + m, j_index + n])
                    n += 1
                m += 1

            # Find the minimum pixel value among the collected values
            pixel_value = min(pixel_values)
            # Update the result image with the minimum pixel value
            imgRes_l[i][j] = pixel_value
            # Move to the next column
            j += 1
        # Move to the next row
        i += 1
    return imgRes_l


#~~~Dilate~~~~~~~~~~~~~~~~~~~~
"""Dilatation: effectue un « ou » logique entre les voisins d’un pixel (augmente l’épaisseur d’un
contour) """
def dilate_cross(img, sizeDilate):
    # Initialize the loop variable for the row index
    i = 0
    # Iterate over the rows of the image
    while i < height:
        # Initialize the loop variable for the column index
        j = 0
        # Iterate over the columns of the image
        while j < width:
            # Handle boundary conditions by replicating edge pixels
            i_index = min(max(i, sizeDilate), height - sizeDilate - 1)
            j_index = min(max(j, sizeDilate), width - sizeDilate - 1)

            # Perform logical OR operation between neighboring pixels based on the kernel
            pixel_values = []
            m = -sizeDilate
            # Iterate over the elements of the horizontal and vertical lines in the kernel
            while m <= sizeDilate:
                # Collect pixel values along the horizontal line
                pixel_values.append(img[i_index + m, j_index])
                # Collect pixel values along the vertical line
                pixel_values.append(img[i_index, j_index + m])
                m += 1

            # Find the maximum pixel value among the collected values
            pixel_value = np.max(np.array(pixel_values))
            # Update the result image with the maximum pixel value
            imgRes_l[i][j] = pixel_value
            # Move to the next column
            j += 1
        # Move to the next row
        i += 1
    return imgRes_l


        
def dilate_rect(img, sizeDilate):
    # Initialize the loop variable for the row index
    i = 0
    # Iterate over the rows of the image
    while i < height:
        # Initialize the loop variable for the column index
        j = 0
        # Iterate over the columns of the image
        while j < width:
            # Handle boundary conditions by replicating edge pixels
            i_index = min(max(i, sizeDilate), height - sizeDilate - 1)
            j_index = min(max(j, sizeDilate), width - sizeDilate - 1)

            # Perform logical OR operation between neighboring pixels based on the kernel
            pixel_values = []
            m = -sizeDilate
            # Iterate over the elements of the square kernel
            while m <= sizeDilate:
                n = -sizeDilate
                while n <= sizeDilate:
                    # Collect pixel values within the square region of the kernel
                    pixel_values.append(img[i_index + m, j_index + n])
                    n += 1
                m += 1

            # Find the maximum pixel value among the collected values
            pixel_value = max(pixel_values)
            # Update the result image with the maximum pixel value
            imgRes_l[i][j] = pixel_value
            # Move to the next column
            j += 1
        # Move to the next row
        i += 1
    return imgRes_l

#~~~MorphEx~~~~~~~~~~~~~~~~~~~~

def morph_rect_open():
    # Apply erosion using a rectangular structuring element
    imgRes_l = erode_rect(img, sizeMorphEx)
    imgRes_l_np = np.array(imgRes_l, dtype=np.uint8)
    # Apply dilation using a rectangular structuring element
    dilate_rect(imgRes_l_np, sizeMorphEx)

def morph_cross_open():
    # Apply erosion using a cross-shaped structuring element
    imgRes_l = erode_cross(img, sizeMorphEx)
    imgRes_l_np = np.array(imgRes_l, dtype=np.uint8)
    # Apply dilation using a cross-shaped structuring element
    dilate_cross(imgRes_l_np, sizeMorphEx)

def morph_rect_closed():
    # Apply dilation using a rectangular structuring element
    imgRes_l = dilate_rect(img, sizeMorphEx)
    imgRes_l_np = np.array(imgRes_l, dtype=np.uint8)
    # Apply erosion using a rectangular structuring element
    erode_rect(imgRes_l_np, sizeMorphEx)

def morph_cross_closed():
    # Apply dilation using a cross-shaped structuring element
    imgRes_l = dilate_cross(img, sizeMorphEx)
    imgRes_l_np = np.array(imgRes_l, dtype=np.uint8)
    # Apply erosion using a cross-shaped structuring element
    erode_cross(imgRes_l_np, sizeMorphEx)

#--Additional Filters------------------------------------------------------------------------------------------------------------------------------------------------------
#~~~Sobel Filter~~~~~~~~~~~~~~~~~~~~
def filter2D_sobel(kernel):
    img_filter = []
    img_filter = np.zeros_like(img) 
    # Initialize the loop variable for the row index
    i = 0
    # Iterate over the rows of the image
    while i < height:
        # Initialize the loop variable for the column index
        j = 0
        # Iterate over the columns of the image
        while j < width:
            # Handle boundary conditions by replicating edge pixels
            i_index = min(max(i, 1), height - 2)
            j_index = min(max(j, 1), width - 2)

            # Perform 2D convolution using the provided kernel
            pixel_value = (
                img[i_index][j_index] * kernel[1][1] +  # Center
                img[i_index - 1][j_index] * kernel[0][1] +  # Left
                img[i_index + 1][j_index] * kernel[2][1] +  # Right
                img[i_index][j_index - 1] * kernel[1][0] +  # Up
                img[i_index][j_index + 1] * kernel[1][2] +  # Down
                img[i_index - 1][j_index - 1] * kernel[0][0] +  # Upper Left
                img[i_index - 1][j_index + 1] * kernel[0][2] +  # Lower Left
                img[i_index + 1][j_index - 1] * kernel[2][0] +  # Upper Right
                img[i_index + 1][j_index + 1] * kernel[2][2]    # Lower Right
            )
            # Update the result image with the computed pixel value
            img_filter[i][j] = pixel_value
            # Move to the next column
            j += 1
        # Move to the next row
        i += 1
    return img_filter

def sobel_filter():
    # Sobel filter kernels
    sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Compute gradients in both x and y directions
    gradient_x = filter2D_sobel(sobel_kernel_x)
    gradient_y = filter2D_sobel(sobel_kernel_y)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize and convert to uint8
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)

    # Update the result image with the Sobel filter output
    i = 0
    while i < height:
        j = 0
        while j < width:
            imgRes_l[i][j] = gradient_magnitude[i][j]
            j += 1
        i += 1
# #~~~Emboss~~~~~~~~~~~~~~~~~~~~
def emboss_filter():
        # Emboss filter kernel
    emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    filter2D(emboss_kernel)


#-----Global Filter Function---------------------------------------------------------------------------------------------------------------------------------------------------

def filter():
    global imgRes_l, th
    # Initialize imgRes_l
    rows, cols = img.shape
    imgRes_l = np.zeros((rows, cols), dtype=np.uint8)
    
    if method == 'laplacien': 
        # Filtre Laplacian
        kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        filter2D(kernel)
    elif method == 'gaussien':
        # filtre gaussien
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        kernel = kernel / 16
        filter2D(kernel)
    elif method == 'erode':
        if structure == "rect":
            erode_rect(img, sizeErode)
        elif structure == "cross":
            erode_cross(img, sizeErode)
    elif method == 'dilate':
        if structure == "rect":
            dilate_rect(img, sizeDilate)
        elif structure == "cross":
            dilate_cross(img, sizeDilate)
    elif method == 'morphex':
        if structure == "rect":
            if morph_type == 'open':
                morph_rect_open()
            elif morph_type == 'close':
                morph_rect_closed()
        elif structure == "cross":
            if morph_type == 'open':
                morph_cross_open()
            elif morph_type == 'close':
                morph_cross_closed()
    elif method == 'sobel':
        sobel_filter()
    elif method == 'emboss':
        emboss_filter() 

    # Thresholding
    thresholding()

    # Convert the result to a NumPy array for displaying
    imgRes_l_np = np.array(imgRes_l, dtype=np.uint8)
    cv2.imshow('result_l', imgRes_l_np)


#--Run------------------------------------------------------------------------------------------------------------------------------------------------------

# Initialize global variables
th = 130
type = 2
sizeDilate = 1
sizeErode = 1
sizeMorphEx = 1

# parameters
method = 'gaussien'
structure = 'rect'
morph_type = 'open'

label_values = ["gaussien", "laplacien", "erode", "dilate","morphex", "sobel", "emboss"]
struct_values = [ "rect", "cross"]
morph_values = [ "open", "close"]

# img = cv2.imread('photo.jpeg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('Filters/Images/img_projet2.jpg', cv2.IMREAD_GRAYSCALE)

height, width = img.shape[:2]

if img is None:
    print("erreur de chargement")
    exit(0)
    
imgRes_l = []


#--CV2 UI------------------------------------------------------------------------------------------------------------------------------------------------------

# Create a window and trackbar
cv2.namedWindow('result_l',  cv2.WINDOW_NORMAL)
cv2.namedWindow('Trackbars',  cv2.WINDOW_NORMAL)
# Set the desired width and height for the window
new_width = 300
new_height = 200

# Resize the window
cv2.resizeWindow('Trackbars', new_width, new_height)

cv2.createTrackbar("Threshold", "Trackbars", th, 255, changeTh)
cv2.createTrackbar("Type", "Trackbars", type, 4, changeType)
cv2.createTrackbar("sizeErode", "Trackbars", sizeErode, 21, change_erode_size)
cv2.createTrackbar("sizeDilate", "Trackbars", sizeDilate, 21, change_dilate_size)
cv2.createTrackbar("sizeMorph", "Trackbars", sizeMorphEx, 21, change_morphEx_size)
cv2.createTrackbar('Filter', 'Trackbars', 0, len(label_values) - 1, changeFilter)
cv2.createTrackbar('Struct', 'Trackbars', 0, len(struct_values) - 1, changeStructure)
cv2.createTrackbar('Morph', 'Trackbars', 0, len(morph_values) - 1, changeMorphType)

filter()



cv2.waitKey(0)
cv2.destroyAllWindows()
