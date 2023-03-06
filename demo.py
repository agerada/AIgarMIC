import cv2
from imutils import contours
import numpy as np
from plate import Plate
import tensorflow as tf

"""
Adapted from https://stackoverflow.com/questions/72089623/how-to-sort-contours-of-a-grid-using-opencv-python
and https://stackoverflow.com/questions/59182827/how-to-get-the-cells-of-a-sudoku-grid-with-opencv"""

INPUT_FILE = 'example_plates/IMG_0033.JPG'
"""
# Load image, grayscale, and simple threshold
image = cv2.imread(INPUT_FILE)
cv2.imshow("original", image)
cv2.waitKey()
blur = cv2.GaussianBlur(image, (25,25), 0)
cv2.imshow('blur', blur)
cv2.waitKey()
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.waitKey()
ret, thresh = cv2.threshold(gray,20,255,cv2.THRESH_BINARY_INV)
cv2.imshow('Binary thresholding', thresh)
cv2.waitKey()

# Adaptive thresholding compensates for lighting differences, but not ideal for this use
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 107, 6)
cv2.imshow('adaptive', adaptive)
cv2.waitKey()

# Otsu algorithm tries to automate the process, but does not seem to work well for this
blur = cv2.GaussianBlur(gray, (5,5), 0) 
ret_otsu, thresh_otsu = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('otsu', thresh_otsu)
cv2.waitKey()


# Find contours and filter using area
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
grid_contours = []
for c in cnts:
    area = cv2.contourArea(c)
    if area > 20000: 
    #if area > 20000 and area < 30000: 
        grid_contours.append(c)
        cv2.drawContours(image, [c], 0, (0,255,0), 3)
        cv2.imshow("contour_test", image)
        cv2.waitKey()
cv2.imshow("contours", image)
cv2.waitKey()
# sort contours and remove biggest (outer) grid square
grid_contours = sorted(grid_contours, key=cv2.contourArea)
grid_contours = grid_contours[:-1]

# Check that we found 96 boxes 
if len(grid_contours) != 96: 
    raise(ValueError)

# Sort contours, starting left to right
(grid_contours, _) = contours.sort_contours(grid_contours, method="left-to-right")
sorted_grid = []
col = [] # temporary list to hold columns while sorting

for (i, c) in enumerate(grid_contours, 1): 
    col.append(c)
    if i % 8 == 0: 
        # found column - sort top to bottom and add to output
        (c_tmp, _) = contours.sort_contours(col, method="top-to-bottom")
        sorted_grid.append(c_tmp)
        col = []

# Iterate through each box
for col in sorted_grid:
    for c in col:
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, (255,255,255), -1)
        result = cv2.bitwise_and(image, mask)
        result[mask==0] = 255
        cv2.imshow('result', result)
        cv2.waitKey(100)

cv2.destroyAllWindows()
"""


# Test out Plate class
test_plate = Plate("test", 0.5, INPUT_FILE)
print(test_plate)

model = tf.keras.models.load_model("models/")
class_names = ['No growth','Poor growth','Good growth']

test_plate.link_model(model, class_names)
test_plate.annotate_images()
print("Overall interpretation of plate: ")
test_plate.print_matrix()
cv2.imshow(test_plate.drug + str(test_plate.concentration), test_plate.image)
cv2.waitKey()
