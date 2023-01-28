import cv2
from imutils import contours
import numpy as np

"""
Adapted from https://stackoverflow.com/questions/72089623/how-to-sort-contours-of-a-grid-using-opencv-python
and https://stackoverflow.com/questions/59182827/how-to-get-the-cells-of-a-sudoku-grid-with-opencv"""
# Load image, grayscale, and simple threshold
image = cv2.imread('images/0.125.jpg')
cv2.imshow("original", image)
cv2.waitKey()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)
cv2.imshow('thres', thresh)
cv2.waitKey()

# Find contours and filter using area
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
grid_contours = []
for c in cnts:
    area = cv2.contourArea(c)
    if area > 20000 and area < 30000: 
        grid_contours.append(c)
        cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
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
