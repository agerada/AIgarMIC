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
from keras.regularizers import L2
from file_handlers import create_dataset_from_directory, predict_images_from_directory

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 160
BATCH_SIZE = 64

def main(): 
    parser = argparse.ArgumentParser("""
    This script loads images from annotations directory and trains ML
    model. Loading from pickled data is not yet implemented
    """)
    parser.add_argument("annotations", type=str, help="Directory containing annotated images")
    parser.add_argument("-p", "--pickled", action="store_true", help="Load data from pickled (.p) files - NOT IMPLEMENTED")
    parser.add_argument("-v", "--visualise", action="store_true", help="Generate visualisations for model diagnostics")
    parser.add_argument("-s", "--save", type=str, help="If specified, tensorflow model will be saved to this folder")
    parser.add_argument("-l", "--log", action="store_true", help="Store performance log in output folder")
    parser.add_argument("-t", "--test_dataset", type=str, help="Testing dataset for final model evaluation. Ideally unseen data. If not provided then input directory is used (whole dataset).")
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
        
        train_dataset,val_dataset = create_dataset_from_directory(annotated_images, image_width=IMAGE_WIDTH, 
                                                                  image_height=IMAGE_HEIGHT)
        
        full_dataset = tf.keras.utils.image_dataset_from_directory(annotated_images, image_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                                                   batch_size=BATCH_SIZE, label_mode="binary")
        
        class_names = train_dataset.class_names
        
        num_classes = len(class_names)

        data_augmentation = Sequential([
            layers.RandomFlip("horizontal",
                            input_shape=(IMAGE_WIDTH,
                                        IMAGE_WIDTH,
                                        3)),
            layers.RandomRotation(0.1),
        ])
        
        # setting bias to compensate for class imbalance
        # needs counts of each class total first
        full_dataset_unbatched = tuple(full_dataset.unbatch())
        labels = [0,0]
        for (_,label) in full_dataset_unbatched:
            labels[int(label.numpy())] += 1
        neg = labels[0]
        pos = labels[1]
        initial_bias = np.log([pos / neg])
        initial_bias = tf.keras.initializers.Constant(initial_bias)

        growth_no_growth = [ 
            # Current working model for first line

            data_augmentation, 
            layers.Rescaling(1./255, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
            layers.Conv2D(32, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),

            layers.Conv2D(32, (3,3), kernel_initializer='he_uniform', activation='relu'), 
            layers.MaxPooling2D((2,2)), 

            layers.Conv2D(64, (3,3), kernel_initializer='he_uniform', activation='relu'), 
            layers.MaxPooling2D((2,2)), 

            layers.Flatten(),
            layers.Dense(64, activation='relu'), 
            layers.Dropout(0.5), 
            layers.Dense(1, activation='sigmoid', bias_initializer = initial_bias)
        ]
        
        growth_poor_growth = [ 
            
            # Working model for second line
            #data_augmentation,
            layers.Rescaling(1./255, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'), 
            layers.Dropout(0.1), 
            layers.Dense(128, activation='relu'), 
            layers.Dropout(0.1), 
            layers.Dense(1, activation='sigmoid', bias_initializer = initial_bias)
        ]

        # Weights adjusted to compensate for class imbalance
        weight_0 = (1 / neg) * ( (neg + pos) / 2)
        weight_1 = (1 / pos) * ( (neg + pos) / 2)
        weights = {0: weight_0, 1: weight_1}

        model = Sequential(growth_poor_growth)
        with tf.device('/device:GPU:0'):
            model.compile(optimizer=keras.optimizers.legacy.RMSprop(learning_rate=.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

            model.summary()

            epochs=300
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs, 
                class_weight=weights
            )
        
        results = model.evaluate(train_dataset, batch_size=BATCH_SIZE, verbose=0)
        print("Loss: {:0.4f}".format(results[0]))

        test_directory = args.test_dataset if args.test_dataset else annotated_images
        annotation_log = predict_images_from_directory(test_directory, model, class_names, IMAGE_WIDTH, IMAGE_HEIGHT)

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