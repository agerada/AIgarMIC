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
from plate import Plate, PlateSet
import tensorflow as tf
import argparse
import random
from utils import convertCV2toKeras, get_conc_from_path, get_paths_from_directory
from multiprocessing import Pool
import csv
from model import SoftmaxModel, BinaryModel, BinaryNestedModel
import sys

def main(): 
    MODEL_IMAGE_X = 160
    MODEL_IMAGE_Y = 160
    SUPPORTED_MODEL_TYPES = ['softmax', 'binary']
    parser = argparse.ArgumentParser(description="Main script to interpret agar dilution MICs",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('directory', type=str, help=
    """
    Directory containing images to process, arranged in subfolders by antibiotic name, e.g.,: \n
    \t directory/ \n
    \t \t antibiotic1_name/ \n
    \t \t \t 0.jpg \n
    \t \t \t 0.125.jpg \n
    \t \t \t 0.25.jpg \n
    """)
    parser.add_argument("-m", "--model", type=str, nargs="*", help="Specify one or more directories containing tensorflow model/s for image classificaion")
    parser.add_argument("-t", "--type_model", type=str, default="softmax", help="Type of keras model, e.g., binary, softmax [default] = softmax")
    parser.add_argument("-o", "--output_file", type=str, help="Specify output file for csv report (will be overwritten)")
    parser.add_argument("-s", "--suppress_validation", action='store_true', help="Suppress manual validation prompts for annotations that have poor accuracy")
    parser.add_argument("-c", "--check_contours", action="store_true", help="Check contours visually")
    args = parser.parse_args()

    plate_images_paths = get_paths_from_directory(args.directory)
    if args.check_contours: 
        cv2.startWindowThread()
        for abx, paths in plate_images_paths.items(): 
            for path in paths: 
                _image = cv2.imread(path)
                try: 
                    split_by_grid(_image, visualise_contours=True, plate_name=abx + '_' + str(get_conc_from_path(path)))
                except ValueError as err:
                    print(err)

        pos_replies = ['y','yes','ye']
        neg_replies = ['n', 'no']
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        while True: 
            input_key = input("""Completed contour checks. Would you like to continue with annotation? [Y / N]
                              Please only proceed if all images have correctly identified 96 boxes!
                              """)
            input_key = input_key.lower()
            if input_key in neg_replies:
                sys.exit()
            elif input_key in pos_replies: 
                print("Continuing with annotation..")
                break
            else:
                print("Unable to recognise input, please try again..")
                continue

    if args.type_model not in SUPPORTED_MODEL_TYPES: 
        sys.exit(f"Model type specified is not supported, please use one of {SUPPORTED_MODEL_TYPES}")

    if args.type_model == "binary" and len(args.model) != 2: 
        sys.exit(
            """
            For a binary type model, need paths to two keras models (first line i.e., no growth vs growth
            and second line i.e., poor growth vs good growth)
            """
        )

    if args.type_model == 'softmax' and len(args.model) != 1: 
        sys.exit(
            """
            Softmax model can only run with one keras model
            """
        )

    if args.type_model == 'softmax':
        class_names = ['No growth','Poor growth','Good growth']
        # Since args.model is a list, unlist
        [path_to_model] = args.model
        model = SoftmaxModel(path_to_model, class_names, trained_x=MODEL_IMAGE_X, trained_y=MODEL_IMAGE_Y)

    elif args.type_model == 'binary': 
        class_names_first_line = ['No growth', 'Growth']
        class_names_second_line = ['Poor growth', 'Good growth']
        first_line_model = BinaryModel(args.model[0], class_names_first_line, trained_x=MODEL_IMAGE_X, trained_y=MODEL_IMAGE_Y)
        second_line_model = BinaryModel(args.model[1], class_names_second_line, trained_x=MODEL_IMAGE_X, trained_y=MODEL_IMAGE_Y)
        model = BinaryNestedModel(first_line_model, second_line_model, first_model_accuracy_acceptance=0.6)

    else: 
        sys.exit(f"Model type specified is not supported, please use one of {SUPPORTED_MODEL_TYPES}")

    abx_superset = {}
    for abx, paths in plate_images_paths.items(): 
        _plates = []
        for path in paths: 
            plate = Plate(abx, get_conc_from_path(path), path, visualise_contours=False, model=model)
            plate.annotate_images()
            _plates.append(plate)
        abx_superset[abx] = _plates

    plateset_list = []
    for abx, plates in abx_superset.items(): 
        plateset_list.append(PlateSet(plates))
    
    for plateset in plateset_list:
        if not args.suppress_validation: 
            plateset.review_poor_images(save_dir = "new_annotations", threshold=.6)
        plateset.calculate_MIC()
        plateset.generate_QC()

    if args.output_file: 
        output_data = []
        for plateset in plateset_list: 
            output_data = output_data + plateset.get_csv_data()
        with open(args.output_file, 'w') as csvfile:
            output_writer = csv.writer(csvfile)
            writer = csv.DictWriter(csvfile, output_data[0].keys())
            writer.writeheader()
            writer.writerows(output_data)

if __name__ == "__main__": 
    main()