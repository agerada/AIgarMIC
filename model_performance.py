#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	model_performance.py
# Author: 	Alessandro Gerada
# Date: 	2023-03-16
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Documentation"""

from utils import get_paths_from_directory
from model import BinaryModel
import cv2

PATH = 'annotations/testing_datasets/growth_no_growth'
MODEL = 'models_binary/growth_no_growth_2023-03-17'
MODEL_WIDTH = 160
MODEL_HEIGHT = 160

results = []

images = get_paths_from_directory(PATH)
model = BinaryModel(MODEL, ['No growth', 'Growth'], trained_x=MODEL_WIDTH, 
                    trained_y=MODEL_HEIGHT)
for i, paths in images.items(): 
    for path in paths: 
        image = cv2.imread(path)
        p = model.predict(image)
        p['true_class'] = int(i)
        results.append(p)

errors = 0
for i in results: 
    if i['growth_code'] != i['true_class']: 
        errors += 1 
    print(f"Prediction={i['prediction']:.2f} \t Growth code={i['growth_code']} \t True class={i['true_class']} \t Accuracy={i['accuracy']:.2f}")

print()
print(f"Errors: {errors} from {len(results)} images ({errors/len(results)*100:.2f}%)")

threshold = 0.9
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
n = len(results)
for i in results: 
    if i['growth_code'] != i['true_class']: 
        #error (positive)
        if i['accuracy'] < threshold: 
            true_positives += 1
        else: 
            false_negatives += 1
    else: 
        #no error (negative)
        if i['accuracy'] < threshold: 
            false_positives += 1
        else: 
            true_negatives += 1

print(f"TP={true_positives} ({true_positives/n*100:.2f}%)")
print(f"FP={false_positives} ({false_positives/n*100:.2f}%)")
print(f"TN={true_negatives} ({true_negatives/n*100:.2f}%)")
print(f"FN={false_negatives} ({false_negatives/n*100:.2f}%)")