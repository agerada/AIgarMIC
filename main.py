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
from utils import convertCV2toKeras

def main(): 
    parser = argparse.ArgumentParser(description="Main script to interpret agar dilution MICs")
    parser.add_argument("-m", "--model", type=str, help="Specify folder containing tensorflow model for image classificaion")
    args = parser.parse_args()

    folder = "images/gent"
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

    if args.model: 
        model = tf.keras.models.load_model(args.model)
        
        
        random_images = []
        predictions = []
        scores = []
        for i in range(10): 
            random_image, _ = random.choice(plates).get_random_colony_image()
            random_images.append(random_image)
            prediction = model.predict(convertCV2toKeras(random_image))
            predictions.append(prediction)
            score = tf.nn.softmax(prediction)
            scores.append(score)
        class_names = ['No growth','Poor growth','Good growth']
        for i,p,s in zip(random_images, predictions, scores): 
            print(f"This image was predicted as: {class_names[np.argmax(s)]} with a prediction of {100 * np.max(s)}")
            cv2.imshow('image', i)
            cv2.waitKey()
        
    class_names = ['No growth','Poor growth','Good growth']

    test_plate = plates[0]
    test_plate.link_model(model, key=class_names)
    print(test_plate.annotate_images())

    for i in plates: 
        i.link_model(model, key=class_names)
        i.annotate_images()
    plate_set = PlateSet(plates)
    print(plate_set.calculate_MIC())
    print(plate_set.convert_mic_matrix(str))
    print(plate_set.generate_QC())
    print()
if __name__ == "__main__": 
    main()