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
from utils import convertCV2toKeras, get_conc_from_path
from multiprocessing import Pool
import csv

def main(): 
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
    args = parser.parse_args()

    class_names = ['No growth','Poor growth','Good growth']

    abx_names = [i for i in os.listdir(args.directory) 
                 if not i.startswith('.') and 
                 os.path.isdir(os.path.join(args.directory,i))]
    
    plate_images_paths = {}
    for abx in abx_names: 
        _path = os.path.join(args.directory, abx)
        _temp_plate_images_paths = os.listdir(_path)
        _temp_plate_images_paths = [i for i in _temp_plate_images_paths if i.count('.jpg') > 0]
        _temp_plate_images_paths = [os.path.join(args.directory,abx,i) for i in _temp_plate_images_paths]
        plate_images_paths[abx] = _temp_plate_images_paths
    
    abx_superset = {}
    model = tf.keras.models.load_model(args.model)
    for abx, paths in plate_images_paths.items(): 
        _plates = []
        for path in paths: 
            plate = Plate(abx, get_conc_from_path(path), path)
            plate.link_model(model, class_names)
            plate.annotate_images()
            _plates.append(plate)
        abx_superset[abx] = _plates

    plateset_list = []
    for abx, plates in abx_superset.items(): 
        plateset_list.append(PlateSet(plates))
    
    for plateset in plateset_list:
        plateset.review_poor_images(save_dir = "new_annotations")
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