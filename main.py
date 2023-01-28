#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	main.py
# Author: 	Alessandro Gerada
# Date: 	2023-01-27
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Script to process images"""

import cv2
from imutils import contours
import numpy as np
from process_plate_image import *
import os
from plate import Plate

def main(): 
    folder = "images/gent"
    gent_images = os.listdir(folder)
    plates = []
    for path in gent_images: 
        try: 
            plates.append(Plate("gentamicin", 0, os.path.join(folder, path)))
        except: 
            print(path, "failed")

if __name__ == "__main__": 
    main()