#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	train_model.py
# Author: 	Alessandro Gerada
# Date: 	2023-01-29
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""This script trains an image annotation model"""

import pickle, os, cv2 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import pathlib
import warnings
import csv
from utils import convertCV2toKeras

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def create_dataset_from_directory(path, val_split = 0.2, 
                                  image_size = (160,160), 
                                  filter_predicate = lambda img,x: True): 
    image_width,image_height = image_size
    train_dataset = tf.keras.utils.image_dataset_from_directory(
                path, 
                validation_split=val_split, 
                subset='training', 
                seed=12345, 
                image_size=(image_width, image_height), 
                batch_size=32, 
                label_mode='binary'
            )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        path, 
        validation_split=val_split, 
        subset='validation', 
        seed=12345, 
        image_size=(image_width, image_height), 
        batch_size=32, 
        label_mode='binary'
    )
    print(f"Found the following labels/classes: {train_dataset.class_names}")
    return train_dataset,val_dataset

def predict_images_from_directory(path, model, class_names, image_width, image_height, threshold = 0.5):
    output = []
    file_paths = {i: os.listdir(os.path.join(path, i)) for i in class_names}
    # add subdirectories 
    file_paths = {i: [os.path.join(path,i,j) for j in file_paths[i] if j.count(".jpg") > 0] for i in file_paths}

    for i in file_paths: 
        for j in file_paths[i]: 
            image = cv2.imread(j)
            prediction = model.predict(convertCV2toKeras(image, image_width, image_height))
            [prediction] = prediction.reshape(-1)
            predicted_class = class_names[0] if prediction <= threshold else class_names[1]
            true_class = i
            path = j
            output.append({"image": image, 
                           "path": path, 
                           "prediction": prediction, 
                           "predicted_class": predicted_class, 
                           "true_class": true_class})
    return output

def main(): 
    parser = argparse.ArgumentParser("""
    This script loads images from annotations directory and trains ML
    model. Loading from pickled data is not yet implemented
    """)
    parser.add_argument("annotations", type=str, help="directory containing annotated images")
    parser.add_argument("-p", "--pickled", action="store_true", help="Load data from pickled (.p) files - NOT IMPLEMENTED")
    parser.add_argument("-v", "--visualise", action="store_true", help="Generate visualisations for model diagnostics")
    parser.add_argument("-s", "--save", type=str, help="If specified, tensorflow model will be saved to this folder")
    parser.add_argument("-l", "--log", action="store_true", help="Store performance log in output folder")
    args = parser.parse_args()

    ANNOTATIONS_FOLDER = args.annotations
    TRAIN_SPLIT = 0.8
    VALIDATION_SPLIT = 0.2

    # load annotation pickles from annotations/ folder
    if args.pickled: 
        annotation_pickles = os.listdir(ANNOTATIONS_FOLDER)
        annotation_pickles = [path for path in annotation_pickles if path.rfind(".p") > 0]
        image_data = []
        annotation_data = []
        for p in annotation_pickles: 
            pickle_path = os.path.join(ANNOTATIONS_FOLDER, p)
            with open(pickle_path, "rb") as f: 
                temp_images = pickle.load(f)
                temp_annotations = pickle.load(f)
            for i in temp_images: 
                image_data.append(i)
            for a in temp_annotations: 
                annotation_data.append(a)
        raise NotImplementedError("Loading from pickled data is not yet implemented")

    else: 
        annotated_images = pathlib.Path(ANNOTATIONS_FOLDER)
        print(f"Number of .jpg files{len(list(annotated_images.glob('*/*.jpg')))}")
        
        image_height = 160
        image_width = 160
        train_dataset,val_dataset = create_dataset_from_directory(annotated_images)
        class_names = train_dataset.class_names
        
        num_classes = len(class_names)

        data_augmentation = Sequential([
            layers.RandomFlip("horizontal",
                            input_shape=(image_height,
                                        image_width,
                                        3)),
            layers.RandomRotation(0.1),
            #layers.RandomZoom(0.1),
        ])
        
        """
        model = Sequential([
        #data_augmentation, 
        layers.Rescaling(1./255, input_shape=(image_height, image_width, 1)),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        #layers.Dropout(0.25),

        #layers.Conv2D(64, (3,3), activation='relu'),
        #layers.MaxPooling2D(),
        #layers.Dropout(0.25),

        #layers.Conv2D(128, (3,3), activation='relu'),
        #layers.MaxPooling2D(),
        #layers.Dropout(0.2), 

        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(num_classes)
        ])
        """
        
        
        growth_no_growth = [ 
            layers.Rescaling(1./255, input_shape=(image_height, image_width, 3)),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'), 
            layers.Dropout(0.1), 
            layers.Dense(64, activation='relu'), 
            layers.Dropout(0.1), 
            layers.Dense(1, activation='sigmoid')
        ]
        
        growth_poor_growth = [ 
            layers.Rescaling(1./255, input_shape=(image_height, image_width, 3)),
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'), 
            layers.Dropout(0.1), 
            layers.Dense(128, activation='relu'), 
            layers.Dropout(0.1), 
            layers.Dense(1, activation='sigmoid')
        ]

        model = Sequential(growth_poor_growth)
        
        model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

        model.summary()

        epochs=40
        history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs, 
        #class_weight={0: 0.5, 1: 1}
        )
        
        annotation_log = predict_images_from_directory(annotated_images, model, class_names, image_width, image_height)

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
                log_file = os.path.join(args.save, "log.csv")
                with open(log_file, "w") as file: 
                    writer = csv.DictWriter(file, ['path', 'prediction','predicted_class','true_class'], 
                                            extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(annotation_log)
            
if __name__ == "__main__": 
    main()