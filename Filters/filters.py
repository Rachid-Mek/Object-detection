import cv2
import numpy as np

#--TrackBar functions------------------------------------------------------------------------------------------------------------------------------------------------------

def changeTh(value):
    global th
    th = value
    filter()
    
def changeType(x):
    global type
    type = x
    filter()
    
def changeFilter(value):
    global method
    method = label_values[value]
    filter()

def changeStructure(value):
    global structure
    structure = struct_values[value]
    filter()
    
def changeMorphType(value):
    global morph_type
    morph_type = morph_values[value]
    filter()
    
def change_erode_size(x):
    global sizeErode
    sizeErode = x 
    filter()
    
def change_dilate_size(x):
    global sizeDilate
    sizeDilate = x 
    filter()

def change_morphEx_size(x):
    global sizeMorphEx
    sizeMorphEx = x 
    filter()
#--Filter Functions TP5------------------------------------------------------------------------------------------------------------------------------------------------------

def treshholding():
    i = 0
    while i < height:
        j = 0
        while j < width:
            if type == 0: #THRESH_BINARY
                imgRes_l[i][j] = 255 if imgRes_l[i][j] > th else 0
                j += 1
            if type == 1: #THRESH_BINARY_INV
                imgRes_l[i][j] = 0 if imgRes_l[i][j] > th else 255
                j += 1
            if type == 2: #THRESH_TRUNC
                imgRes_l[i][j] = th if imgRes_l[i][j] > th else imgRes_l[i][j]
                j += 1
            if type == 3: #THRESH_TOZERO
                imgRes_l[i][j] = imgRes_l[i][j] if imgRes_l[i][j] > th else 0
                j += 1
            if type == 4: #THRESH_TOZERO_INV
                imgRes_l[i][j] = 0 if imgRes_l[i][j] > th else imgRes_l[i][j]
                j += 1
        i += 1
        
def filter2D(kernel):
        i = 0
        while i < height:
            j = 0
            while j < width:
                i_index = min(max(i,1), height-2)
                j_index = min(max(j,1), width-2)

                pixel_value = (
                    img[i_index][j_index] * kernel[1][1] #center
                    + img[i_index - 1][j_index] * kernel[0][1] #left
                    + img[i_index + 1][j_index] * kernel[2][1] #right
                    
                    + img[i_index][j_index - 1] * kernel[1][0] # up
                    + img[i_index][j_index + 1] * kernel[1][2] #down
                    
                    + img[i_index - 1][j_index - 1] * kernel[0][0] #upper left
                    + img[i_index - 1][j_index + 1] * kernel[0][2] #lower left
                    + img[i_index + 1][j_index - 1] * kernel[2][0] #upper ri_indexght
                    + img[i_index + 1][j_index + 1] * kernel[2][2] #lower right
                )
                imgRes_l[i][j] = pixel_value
                j += 1
            i += 1
            
#--Filter Functions TP9------------------------------------------------------------------------------------------------------------------------------------------------------
#~~~Erode~~~~~~~~~~~~~~~~~~~~
""" Erosion: effectue un « et » logique entre les voisins d’un pixel (diminue le contour de l’ordre
d’un pixel)"""

def erode_cross(sizeErode):
    kernel = np.zeros((sizeErode*2+1, sizeErode*2+1), dtype=np.uint8)
    mid = sizeErode
    kernel[mid, :] = 1  # Horizontal line
    kernel[:, mid] = 1  # Vertical line
    print('cross')
    print(kernel)
    
    i = 0
    while i < height:
        j = 0
        while j < width:
            # Handle boundary conditions by replicating edge pixels
            i_index = min(max(i, sizeErode), height - sizeErode - 1)
            j_index = min(max(j, sizeErode), width - sizeErode - 1)

            # Perform logical AND operation between neighboring pixels based on the kernel
            upper_line = img[i_index - sizeErode, j_index - sizeErode:j_index + sizeErode + 1]
            lower_line = img[i_index + sizeErode, j_index - sizeErode:j_index + sizeErode + 1]
            left_line = img[i_index - sizeErode:i_index + sizeErode + 1, j_index - sizeErode]
            right_line = img[i_index - sizeErode:i_index + sizeErode + 1, j_index + sizeErode]

            pixel_value = (
                np.min(np.concatenate((upper_line, lower_line, left_line, right_line)))
            )
            imgRes_l[i][j] = pixel_value
            j += 1
        i += 1

        
def erode_rect(sizeErode):
    kernel = np.ones((sizeErode*2+1, sizeErode*2+1), dtype=np.uint8)
    print('rect')
    print(kernel)
    
    i = 0
    while i < height:
        j = 0
        while j < width:
            # Handle boundary conditions by replicating edge pixels
            i_index = min(max(i, sizeErode), height - sizeErode - 1)
            j_index = min(max(j, sizeErode), width - sizeErode - 1)

            # Perform logical AND operation between neighboring pixels based on the kernel
            pixel_value = (
                (img[i_index - sizeErode:i_index + sizeErode + 1, j_index - sizeErode:j_index + sizeErode + 1] * kernel).min()
            )
            imgRes_l[i][j] = pixel_value
            j += 1
        i += 1

#~~~Dilate~~~~~~~~~~~~~~~~~~~~
"""Dilatation: effectue un « ou » logique entre les voisins d’un pixel (augmente l’épaisseur d’un
contour) """
def dilate_cross(sizeDilate):
    kernel = np.zeros((sizeDilate*2+1, sizeDilate*2+1), dtype=np.uint8)
    mid = sizeDilate
    kernel[mid, :] = 1  # Horizontal line
    kernel[:, mid] = 1  # Vertical line
    print('cross')
    print(kernel)
    
    i = 0
    while i < height:
        j = 0
        while j < width:
            # Handle boundary conditions by replicating edge pixels
            i_index = min(max(i, sizeDilate), height - sizeDilate - 1)
            j_index = min(max(j, sizeDilate), width - sizeDilate - 1)

            # Perform logical OR operation between neighboring pixels based on the kernel
            upper_line = img[i_index - sizeDilate, j_index - sizeDilate:j_index + sizeDilate + 1]
            lower_line = img[i_index + sizeDilate, j_index - sizeDilate:j_index + sizeDilate + 1]
            left_line = img[i_index - sizeDilate:i_index + sizeDilate + 1, j_index - sizeDilate]
            right_line = img[i_index - sizeDilate:i_index + sizeDilate + 1, j_index + sizeDilate]

            pixel_value = (
                np.max(np.concatenate((upper_line, lower_line, left_line, right_line)))
            )
            imgRes_l[i][j] = pixel_value
            j += 1
        i += 1

        
def dilate_rect(sizeDilate):
    kernel = np.ones((sizeDilate*2+1, sizeDilate*2+1), dtype=np.uint8)
    print('rect')
    print(kernel)
    
    i = 0
    while i < height:
        j = 0
        while j < width:
            # Handle boundary conditions by replicating edge pixels
            i_index = min(max(i, sizeDilate), height - sizeDilate - 1)
            j_index = min(max(j, sizeDilate), width - sizeDilate - 1)

            # Perform logical OR operation between neighboring pixels based on the kernel
            pixel_value = (
                (img[i_index - sizeDilate:i_index + sizeDilate + 1, j_index - sizeDilate:j_index + sizeDilate + 1] * kernel).max()
            )
            imgRes_l[i][j] = pixel_value
            j += 1
        i += 1

#~~~MorphEx~~~~~~~~~~~~~~~~~~~~

def morph_rect_open():
    erode_rect(sizeMorphEx)
    dilate_rect(sizeMorphEx)
def morph_cross_open():
    erode_cross(sizeMorphEx)
    dilate_cross(sizeMorphEx)
def morph_rect_closed():
    dilate_rect(sizeMorphEx)
    erode_rect(sizeMorphEx)
def morph_cross_closed():
    dilate_cross(sizeMorphEx)
    erode_cross(sizeMorphEx)
       


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
            erode_rect(sizeErode)
        elif structure == "cross":
            erode_cross(sizeErode)
    elif method == 'dilate':
        if structure == "rect":
            dilate_rect(sizeDilate)
        elif structure == "cross":
            dilate_cross(sizeDilate)
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

    # Thresholding
    treshholding()

    # Convert the result to a NumPy array for displaying
    imgRes_l_np = np.array(imgRes_l, dtype=np.uint8)
    cv2.imshow('result_l', imgRes_l_np)


#--Run------------------------------------------------------------------------------------------------------------------------------------------------------

# Initialize global variables
th = 130
type = 0
sizeDilate = 0
sizeErode = 0
sizeMorphEx = 0

# parameters
method = 'gaussien'
structure = 'rect'
morph_type = 'open'

label_values = ["gaussien", "laplacien", "erode", "dilate","morphex"]
struct_values = [ "rect", "cross"]
morph_values = [ "open", "close"]

img = cv2.imread('Filters/Images/img_projet2.jpg', cv2.IMREAD_GRAYSCALE)
height, width = img.shape[:2]

if img is None:
    print("erreur de chargement")
    exit(0)
    
imgRes_l = []


#--CV2 UI------------------------------------------------------------------------------------------------------------------------------------------------------

# Create a window and trackbar
cv2.namedWindow('result_l',  cv2.WINDOW_NORMAL)
filter()

cv2.createTrackbar("Threshold", "result_l", th, 255, changeTh)
cv2.createTrackbar("Type", "result_l", type, 4, changeType)
cv2.createTrackbar("sizeErode", "result_l", sizeErode, 21, change_erode_size)
cv2.createTrackbar("sizeDilate", "result_l", sizeDilate, 21, change_dilate_size)
cv2.createTrackbar("sizeMorph", "result_l", sizeMorphEx, 21, change_morphEx_size)
cv2.createTrackbar('Filter', 'result_l', 0, len(label_values) - 1, changeFilter)
cv2.createTrackbar('Struct', 'result_l', 0, len(struct_values) - 1, changeStructure)
cv2.createTrackbar('Morph', 'result_l', 0, len(morph_values) - 1, changeMorphType)

cv2.waitKey(0)
cv2.destroyAllWindows()
