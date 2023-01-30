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

def convertCV2toKeras(image): 
    # resize
    image =cv2.resize(image, (160, 160))
    # convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape(1, 160, 160, 3)
    image = image.astype(np.float32)
    return image