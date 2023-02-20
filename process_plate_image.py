#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	process_plate_image.py
# Author: 	Alessandro Gerada
# Date: 	2023-01-27
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""This script contains function/s that does pre-processing of a 96-well agar plate image"""

import cv2
from imutils import contours
import numpy as np

def split_by_grid(image, nrow = 8): 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)

    # Find contours and filter using area
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    grid_contours = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1000: 
        #if area > 20000 and area < 30000: 
            grid_contours.append(c)
            #cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

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
        if i % nrow == 0: 
            # found column - sort top to bottom and add to output
            (c_tmp, _) = contours.sort_contours(col, method="top-to-bottom")
            sorted_grid.append(c_tmp)
            col = []

    out_matrix = [[0 for x in range(len(sorted_grid))] for y in range(nrow)]

    # Iterate through each box
    for j,col in enumerate(sorted_grid):
        for i,c in enumerate(col):
            x,y,w,h = cv2.boundingRect(c)
            cropped_image = image[y:y+h, x:x+w]
            out_matrix[i][j] = cropped_image
            
    return out_matrix
