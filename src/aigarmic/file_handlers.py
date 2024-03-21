#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	file_handlers.py
# Author: 	Alessandro Gerada
# Date: 	2023-08-11
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Functions to facilitate working with image data files"""
import csv
import os
from os import path
from typing import Optional

import cv2
import keras.callbacks
import numpy as np
import tensorflow as tf

from aigarmic.utils import convertCV2toKeras


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


def predict_colony_images_from_directory(path,
                                         model,
                                         class_names,
                                         image_width,
                                         image_height,
                                         model_type,
                                         save_path: Optional[str] = None,
                                         binary_threshold = 0.5):
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
                predicted_class = class_names[0] if prediction <= binary_threshold else class_names[1]
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
    if save_path:
        if not os.path.exists(save_path):
            print(f"Target save directory does not exist: {save_path}")
            print("Skipping saving")
        else:
            annotation_log_file = path.join(save_path, "test_dataset_log.csv")
            with open(annotation_log_file, "w") as file:
                writer = csv.DictWriter(file, ['path', 'prediction', 'predicted_class', 'true_class'],
                                        extrasaction='ignore')
                writer.writeheader()
                writer.writerows(output)
    return output


def save_training_log(model_history: keras.callbacks.History, save_path):
    training_log_file = path.join(save_path, "training_log.csv")
    with open(training_log_file, "w") as file:
        writer = csv.writer(file)
        h = zip(
            model_history.history['accuracy'],
            model_history.history['val_accuracy'],
            model_history.history['loss'],
            model_history.history['val_loss'],
            range(len(model_history.history['accuracy'])))
        writer.writerow(['accuracy', 'val_accuracy', 'loss', 'val_loss', 'epoch'])
        writer.writerows(h)
