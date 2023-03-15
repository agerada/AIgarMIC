#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename:      utils.py
# Author:        Alessandro Gerada
# Date:          01/10/2022
# Copyright:     Alessandro Gerada 2022
# Email:         alessandro.gerada@liverpool.ac.uk

"""
Documentation
"""
import cv2
import numpy as np
import os

def rectContour(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            #print("Corner points", len(approx))
            if len(approx == 4):
                rectCon.append(i)

    rectCon = sorted(rectCon, key = cv2.contourArea, reverse=True)
    return rectCon

def getCornerPoints(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    return approx

def convertCV2toKeras(image, size_x=160, size_y=160): 
    # resize
    image =cv2.resize(image, (size_x, size_y))
    # convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape(1, size_x, size_y, 3)
    image = image.astype(np.float32)
    return image

def get_conc_from_path(path): 
    """
    get concentration from plate image path, e.g.
    antibiotic1/0.125.jpg -> 0.125
    """
    split_text = os.path.split(path)
    split_text = split_text[-1]
    conc_str = os.path.splitext(split_text)[0]
    return float(conc_str)

def get_paths_from_directory(path): 
    """
    Returns a dict of abx_names: [image1_path, image2_path..etc]
    If there are no antibiotic subdirectories, "unnamed" is used 
    for abx_names (length = 1)
    """
    abx_names = [i for i in os.listdir(path) 
                if not i.startswith('.') and 
                os.path.isdir(os.path.join(path,i))]
    
    if not abx_names: 
        abx_names = [""]
    
    plate_images_paths = {}
    for abx in abx_names: 
        _path = os.path.join(path, abx)
        _temp_plate_images_paths = os.listdir(_path)
        _temp_plate_images_paths = [i for i in _temp_plate_images_paths if i.count('.jpg') > 0 or i.count('.JPG') > 0]
        _temp_plate_images_paths = [os.path.join(path,abx,i) for i in _temp_plate_images_paths]
        plate_images_paths[abx] = _temp_plate_images_paths

    return plate_images_paths