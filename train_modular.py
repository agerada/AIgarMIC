#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	train_modular.py
# Author: 	Alessandro Gerada
# Date: 	2023-08-11
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Modular training function that works with binary or softmax"""

import os, cv2 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import pathlib
import warnings
import csv


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from sklearn.utils import class_weight
from tensorflow.keras import initializers
from file_handlers import create_dataset_from_directory, predict_images_from_directory

from nn_design import *

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
BATCH_SIZE = 64
TYPE = "softmax"

def main(): 
    parser = argparse.ArgumentParser("""
    This script loads images from annotations directory and trains ML
    model. Loading from pickled data is not yet implemented
    """)
    parser.add_argument("annotations", type=str, help="Directory containing annotated images")
    parser.add_argument("-v", "--visualise", action="store_true", help="Generate visualisations for model diagnostics")
    parser.add_argument("-s", "--save", type=str, help="If specified, tensorflow model will be saved to this folder")
    parser.add_argument("-l", "--log", action="store_true", help="Store performance log in output folder")
    parser.add_argument("-t", "--test_dataset", type=str, help="Testing dataset for final model evaluation. Ideally unseen data. If not provided then input directory is used (whole dataset).")
    args = parser.parse_args()

    ANNOTATIONS_FOLDER = args.annotations

    annotated_images = pathlib.Path(ANNOTATIONS_FOLDER)
    print(f"Number of .jpg files{len(list(annotated_images.glob('*/*.jpg')))}")
    
    if TYPE=="binary":
        label_mode = "binary"
    elif TYPE=="softmax":
        label_mode = "int"
    else:
        raise ValueError("Unrecognised TYPE")

    train_dataset,val_dataset = create_dataset_from_directory(
        annotated_images,
        label_mode = label_mode,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        batch_size=BATCH_SIZE)
    

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    model = Sequential(keras_model) 

    if TYPE=="softmax":
        # Adjust class weights
        obs_dict = dict(zip(list(range(num_classes)), [0] * num_classes))
        all_class_obs = []
        for i,j in train_dataset:
            for k in j:
                k_arr = k.numpy()
                obs_dict[k_arr] += 1
                all_class_obs.append(k_arr)

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(all_class_obs), y = all_class_obs)
        class_weights = dict(zip(list(range(num_classes)), class_weights))
        initial_bias = initializers.Zeros()
        #class_weights = None
        #initial_bias = None
        model.add(layers.Dense(num_classes, activation='softmax', bias_initializer = initial_bias))

    if TYPE=="binary":
    # Weights adjusted to compensate for class imbalance
    
        # setting bias to compensate for class imbalance
        # needs counts of each class total first
        full_dataset = tf.keras.utils.image_dataset_from_directory(annotated_images, image_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                                            batch_size=BATCH_SIZE, label_mode="binary")
    
        full_dataset_unbatched = tuple(full_dataset.unbatch())
        labels = [0,0]
        for (_,label) in full_dataset_unbatched:
            labels[int(label.numpy())] += 1
        neg = labels[0]
        pos = labels[1]
        initial_bias = np.log([pos / neg])
        initial_bias = tf.keras.initializers.Constant(initial_bias)

        weight_0 = (1 / neg) * ( (neg + pos) / 2)
        weight_1 = (1 / pos) * ( (neg + pos) / 2)
        class_weights = {0: weight_0, 1: weight_1}
        
        model.add(layers.Dense(1, activation='sigmoid', bias_initializer = initial_bias))
        
    with tf.device('/device:GPU:0'):
        if TYPE=="binary":
            model.compile(optimizer=keras.optimizers.legacy.RMSprop(learning_rate=.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
        elif TYPE=="softmax":
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else: 
            raise ValueError("Unknown TYPE")
        model.summary()

        epochs=200
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs, 
            class_weight=class_weights
        )
    
    results = model.evaluate(train_dataset, batch_size=BATCH_SIZE, verbose=0)
    print("Loss: {:0.4f}".format(results[0]))

    test_directory = args.test_dataset if args.test_dataset else annotated_images
    annotation_log = predict_images_from_directory(test_directory, model, class_names, IMAGE_WIDTH, IMAGE_HEIGHT, TYPE)

    if args.visualise: 
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        for i in annotation_log: 
            if i["predicted_class"] != i["true_class"]: 
                print(
                f"This image was misclassified as {i['predicted_class']} "
                f"with prediction of {i['prediction']} "
                f"(should have been {i['true_class']})")
                cv2.imshow(str(i['path']),i['image'])
                cv2.waitKey()
            else: 
                print(f"Correct classification with prediction {i['prediction']} (class {i['true_class']})")

    if args.save: 
        if not os.path.exists(args.save): 
            os.mkdir(args.save)
        model.save(args.save)
        print(f"Model saved to {args.save}")
            
    if args.log: 
        if not args.save: 
            warnings.warn("Unable to save log file because model save path not provided, please use -s to provide path.")
        else: 
            annotation_log_file = os.path.join(args.save, "test_dataset_log.csv")
            with open(annotation_log_file, "w") as file: 
                writer = csv.DictWriter(file, ['path', 'prediction','predicted_class','true_class'], 
                                        extrasaction='ignore')
                writer.writeheader()
                writer.writerows(annotation_log)
            
            training_log_file = os.path.join(args.save, "training_log.csv")
            with open(training_log_file, "w") as file:
                writer = csv.writer(file)
                h = zip(
                    history.history['accuracy'],
                    history.history['val_accuracy'],
                    history.history['loss'],
                    history.history['val_loss'],
                    range(epochs))
                writer.writerow(['accuracy', 'val_accuracy', 'loss', 'val_loss', 'epoch'])
                writer.writerows(h)

if __name__ == "__main__": 
    main()