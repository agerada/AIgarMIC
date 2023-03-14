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

def find_threshold_value(image, start = 20, end = 100, by = 1, 
                   look_for = 96, area_lower_bound = 1000): 
    for i in range(start, end, by): 
        ret, thresh = cv2.threshold(image, i, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours and filter using area
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        grid_contours = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area > area_lower_bound: 
                grid_contours.append(c)

        # sort contours and remove biggest (outer) grid square
        grid_contours = sorted(grid_contours, key=cv2.contourArea)
        grid_contours = grid_contours[:-1]

        # If we find the target boxes, return contours and threshold
        if len(grid_contours) == look_for: 
            return grid_contours, i
    return None, None


def split_by_grid(image, nrow = 8, visualise_contours = False, plate_name = None): 
    if visualise_contours and not plate_name: 
         raise ValueError("Pass plate name to split_by_grid if using visualise_contours")
    blur = cv2.GaussianBlur(image, (25,25), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grid_contours, threshold_value = find_threshold_value(gray)
    
    if visualise_contours: 
                _image = image
                cv2.drawContours(_image, grid_contours, -1, (0,255,0), 10)
                cv2.imshow(plate_name, _image)
                cv2.waitKey()

    if not grid_contours: 
        raise ValueError("Unable to find contours threshold that returns correct number of colony images")
    else: 
        print(f"Using {threshold_value} threshold value")

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
