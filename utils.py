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