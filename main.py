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

def main(): 
    MODEL_IMAGE_X = 160
    MODEL_IMAGE_Y = 160

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
    parser.add_argument("-m", "--model", type=str, help="Specify file containing tensorflow model for image classificaion")
    parser.add_argument("-o", "--output_file", type=str, help="Specify output file for csv report (will be overwritten)")
    parser.add_argument("-s", "--suppress_validation", action='store_true', help="Suppress manual validation prompts for annotations that have poor accuracy")
    parser.add_argument("-c", "--check_contours", action="store_true", help="Check contours visually")
    args = parser.parse_args()

    class_names = ['No growth','Poor growth','Good growth']

    plate_images_paths = get_paths_from_directory(args.directory)
    if args.check_contours: 
        for abx, paths in plate_images_paths.items(): 
            for path in paths: 
                _image = cv2.imread(path)
                split_by_grid(_image, visualise_contours=True, plate_name=abx + '_' + str(get_conc_from_path(path)))

    abx_superset = {}
    model = tf.keras.models.load_model(args.model)
    for abx, paths in plate_images_paths.items(): 
        _plates = []
        for path in paths: 
            plate = Plate(abx, get_conc_from_path(path), path, visualise_contours=False)
            plate.link_model(model, class_names, model_image_x=MODEL_IMAGE_X, model_image_y=MODEL_IMAGE_Y)
            plate.annotate_images()
            _plates.append(plate)
        abx_superset[abx] = _plates

    plateset_list = []
    for abx, plates in abx_superset.items(): 
        plateset_list.append(PlateSet(plates))
    
    for plateset in plateset_list:
        if not args.suppress_validation: 
            plateset.review_poor_images(save_dir = "new_annotations", threshold=.9)
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