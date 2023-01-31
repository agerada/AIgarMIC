#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	manual_annotator.py
# Author: 	Alessandro Gerada
# Date: 	2023-01-28
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""
This script shows random images from 96-well plates, and waits for user input to 
annotate the image. Use the following key: 
0 = no growth
1 = faint growth or colony
2 = normal growth
The output is stored in annotations/
"""

import cv2
from imutils import contours
import numpy as np
from process_plate_image import *
import os
from plate import Plate
import argparse
from random import choice
import pickle
from datetime import datetime

def main(): 
    OUTPUT_DIR = "annotations/"
    parser = argparse.ArgumentParser(description="Manually annotate plate images")
    parser.add_argument('directory', type=str, help="Directory containing plate images")
    parser.add_argument('-o', '--output_to_files', action='store_true', help="Output to .jpg files in subfolders in annotations/    If not used, then defaults to storing in .p pickle files (NOT IMPLEMENTED)  [ default FALSE ]")
    args = parser.parse_args()

    codes = {48: 0, 49: 1, 50: 2, 27: "esc"}

    folder = args.directory
    gent_images = os.listdir(folder)
    gent_images = [i for i in gent_images if i.count('.jpg') > 0]
    plates = []
    for path in gent_images: 
        try: 
            conc = float(os.path.splitext(path)[0])
            plates.append(Plate("gentamicin", conc, os.path.join(folder, path)))
        except Exception as e: 
            print(f"Error while trying to import {path}: ")
            print(e)
    plates.sort(reverse=True)

    print("Use the following key for input: ")
    print("0 = no growth")
    print("1 = faint growth or colony")
    print("2 = normal growth")
    print("Press esc to cancel at any time. ")
    print()

    while True: 
        try: 
            n = int(input("How many images do you want to annotate? "))
        except ValueError: 
            print("Unable to parse input, try again.. ")
            continue
        else: 
            break

    output_images = []
    output_annotations = []
    output_image_codes = []
    for i in range(n): 
        x,code = choice(plates).get_random_colony_image()
        cv2.imshow('image', x)
        while True: 
            input_key = cv2.waitKey()
            if input_key not in codes: 
                continue
            else: 
                break
        if codes[input_key] == 'esc': 
            break
        else: 
            output_images.append(x)
            output_annotations.append(codes[input_key])
            output_image_codes.append(code)

    if args.output_to_files: 
        if not os.path.exists(OUTPUT_DIR): 
            os.mkdir(OUTPUT_DIR)

        # make class folders
        for a in output_annotations: 
            if not os.path.exists(os.path.join(OUTPUT_DIR, str(a))): 
                os.mkdir(os.path.join(OUTPUT_DIR, str(a)))
        
        for i,a,c in zip(output_images, output_annotations, output_image_codes): 
            cv2.imwrite(os.path.join(OUTPUT_DIR, str(a), c + '.jpg'),i)

    else: 
        input_filename = input("Done. Enter filename for annotations (optional).. ")
        input_filename = input_filename + "_" + datetime.now().strftime('%Y-%m-%d_%H%M%S') + ".p"
        
        if not os.path.exists(OUTPUT_DIR): 
            os.mkdir(OUTPUT_DIR)
        output_file_path = os.path.join(OUTPUT_DIR, input_filename)

        with open(output_file_path, "wb") as f: 
            pickle.dump(output_images, f)
            pickle.dump(output_annotations, f)

if __name__ == "__main__": 
    main()