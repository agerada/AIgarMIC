#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	file_handlers.py
# Author: 	Alessandro Gerada
# Date: 	2023-08-11
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Functions to facilitate working with image data files"""
import tensorflow as tf
from utils import convertCV2toKeras
import os, cv2
import numpy as np

def create_dataset_from_directory(path, label_mode, image_width, image_height, val_split = 0.2, 
                                  filter_predicate = lambda img,x: True,
                                  batch_size=32): 
    train_dataset = tf.keras.utils.image_dataset_from_directory(
                path, 
                validation_split=val_split, 
                subset='training', 
                seed=12345, 
                image_size=(image_width, image_height), 
                batch_size=batch_size, 
                label_mode=label_mode
            )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        path, 
        validation_split=val_split, 
        subset='validation', 
        seed=12345, 
        image_size=(image_width, image_height), 
        batch_size=batch_size, 
        label_mode=label_mode
    )
    print(f"Found the following labels/classes: {train_dataset.class_names}")
    return train_dataset,val_dataset

def predict_images_from_directory(path,
                                  model,
                                  class_names,
                                  image_width,
                                  image_height,
                                  model_type,
                                  threshold = 0.5):
    output = []
    file_paths = {i: os.listdir(os.path.join(path, i)) for i in class_names}
    # add subdirectories 
    file_paths = {i: [os.path.join(path,i,j) for j in file_paths[i] if j.count(".jpg") > 0] for i in file_paths}

    for i in file_paths: 
        for j in file_paths[i]: 
            image = cv2.imread(j)
            true_class = i
            path = j
            prediction = model.predict(convertCV2toKeras(image, image_width, image_height))
            if model_type=="binary":
                [prediction] = prediction.reshape(-1)
                predicted_class = class_names[0] if prediction <= threshold else class_names[1]
            elif model_type=="softmax":
                prediction = tf.nn.softmax(prediction)
                prediction_value = np.max(prediction)
                predicted_class = np.argmax(prediction)
                prediction = prediction_value
            output.append({"image": image, 
                           "path": path, 
                           "prediction": prediction, 
                           "predicted_class": predicted_class, 
                           "true_class": true_class})
    return output
