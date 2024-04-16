#!/usr/bin/env python
# Filename: 	manual_annotator.py
# Author: 	Alessandro Gerada
# Date: 	2023-01-28
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""
This script shows random images from 96-well plates, and waits for user input to 
annotate the image. Supports up to 10 growth labels (0 to 9).
The output is stored in annotations/
"""

from aigarmic.file_handlers import get_concentration_from_path, get_paths_from_directory
from aigarmic.plate import Plate
import os
import cv2  # pylint: disable=import-error
import argparse
from random import choice
from datetime import datetime


def manual_annotator_parser():
    parser = argparse.ArgumentParser(description="""
        Manually label random colony images from plates to generate annotated image database.
        Supports labels from 0 to 9. """)
    parser.add_argument('input_directory', type=str,
                        help="Directory containing plate images")
    parser.add_argument('-o', '--output_directory', type=str, default='annotations/',
                        help='Path to store annotation output files')
    return parser


def main(): 
    parser = manual_annotator_parser()
    args = parser.parse_args()
    codes = {}
    for ascii_code, class_code in zip(range(48, 58), range(0, 10)):
        codes[ascii_code] = class_code

    stop_codes = {27: "esc"}
    codes.update(stop_codes)

    input_dir = args.input_directory
    plate_image_paths = get_paths_from_directory(input_dir)
    plates = []
    for abx, paths in plate_image_paths.items(): 
        for path in paths: 
            try: 
                concentration = get_concentration_from_path(path)
                plates.append(Plate(abx, concentration, path, visualise_contours=False,
                                    n_row=8, n_col=12))
            except FileNotFoundError as e:
                print(f"Error while trying to import {path}: ")
                print(e)

    print("Enter a number from 0 to 9 corresponding to growth quality")
    print("Press esc to cancel at any time. ")
    print()

    n = 0
    while True: 
        try: 
            n = int(input("How many images do you want to annotate? "))
        except ValueError: 
            print("Unable to parse input, try again.. ")
            continue
        break

    output_images = []
    output_annotations = []
    output_image_codes = []

    for i in range(n): 
        x, code = choice(plates).get_colony_image()
        code = code + '_' + datetime.now().strftime('%Y-%m-%d_%H%M%S')
        cv2.imshow('image', x)  # pylint: disable=no-member
        print("Please enter classification for this image..")
        while True: 
            input_key = cv2.waitKey()  # pylint: disable=no-member
            if input_key not in codes: 
                print("Input not recognised, please try again..")
                continue
            else: 
                break
        if input_key in stop_codes:
            print("Exiting.")
            break
        else: 
            output_images.append(x)
            output_annotations.append(codes[input_key])
            output_image_codes.append(code)

    output_dir = args.output_directory
    if not os.path.exists(output_dir): 
        os.mkdir(output_dir)

    # make class folders
    for a in output_annotations: 
        if not os.path.exists(os.path.join(output_dir, str(a))): 
            os.mkdir(os.path.join(output_dir, str(a)))
    
    for i, a, c in zip(output_images, output_annotations, output_image_codes):
        cv2.imwrite(os.path.join(output_dir, str(a), c + '.jpg'), i)  # pylint: disable=no-member


if __name__ == "__main__": 
    main()
