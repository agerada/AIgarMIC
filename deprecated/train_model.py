#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	train_model.py
# Author: 	Alessandro Gerada
# Date: 	2023-01-29
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""This script trains an image annotation model"""

from aigarmic import convertCV2toKeras

import pickle, os, cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.utils import class_weight
from tensorflow.keras import initializers

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 160

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
                color_mode='grayscale'
            )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        path, 
        validation_split=val_split, 
        subset='validation', 
        seed=12345, 
        image_size=(image_width, image_height), 
        batch_size=32, 
        color_mode='grayscale'
    )
    print(f"Found the following labels/classes: {train_dataset.class_names}")
    return train_dataset,val_dataset

def main(): 
    parser = argparse.ArgumentParser("""
    This script loads images from annotations directory and trains ML
    model. Loading from pickled data is not yet implemented
    """)
    parser.add_argument("annotations", type=str, help="directory containing annotated images")
    parser.add_argument("-p", "--pickled", action="store_true", help="Load data from pickled (.p) files - NOT IMPLEMENTED")
    parser.add_argument("-v", "--visualise", action="store_true", help="Generate visualisations for model diagnostics")
    parser.add_argument("-s", "--save", type=str, help="If specified, tensorflow model will be saved to this folder")
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
        train_dataset,val_dataset = create_dataset_from_directory(annotated_images, image_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
        class_names = train_dataset.class_names

        if args.visualise: 
            plt.figure(figsize=(10, 10))
            for images, labels in train_dataset.take(1):
                for i in range(9):
                    ax = plt.subplot(3, 3, i + 1)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    plt.title(class_names[labels[i]])
                    plt.axis("off")
        
        num_classes = len(class_names)
        obs_dict = dict(zip(list(range(num_classes)), [0] * num_classes))
        all_class_obs = []
        for i,j in train_dataset:
            for k in j:
                k_arr = k.numpy()
                obs_dict[k_arr] += 1
                all_class_obs.append(k_arr)

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(all_class_obs), y = all_class_obs)
        class_weights = dict(zip(list(range(num_classes)), class_weights))
        
        data_augmentation = Sequential([
            layers.RandomFlip("horizontal",
                            input_shape=(IMAGE_WIDTH,
                                        IMAGE_HEIGHT,
                                        3)),
            layers.RandomRotation(0.1),
            #layers.RandomZoom(0.1),
        ])
        
        model = Sequential([
        #data_augmentation, 
        layers.Rescaling(1./255, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
        
        layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),

        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        #layers.Conv2D(128, (3,3), activation='relu'),
        #layers.MaxPooling2D(),
        #layers.Dropout(0.2), 

        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes, bias_initializer=initializers.Zeros())
        ])
        
        """
        model = Sequential([ 
            #data_augmentation,
            layers.Rescaling(1./255, input_shape=(image_height, image_width, 1)),
            layers.Flatten(),
            layers.Dense(4, activation='relu'), 
            #layers.Dropout(0.25), 
            layers.Dense(4, activation='relu'), 
            #layers.Dropout(0.25), 
            #layers.Dense(16, activation='relu'), 
            layers.Dense(num_classes)
        ])
        """
        
        
        model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=.01),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

        model.summary()

        epochs=200
        #weights = {0: 1., 1: 1., 2: 1.}
        history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs, 
        class_weight=class_weights
        )

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

            file_paths = {i: os.listdir(os.path.join(annotated_images, i)) for i in class_names}
            # add subdirectories 
            file_paths = {i: [os.path.join(annotated_images,i,j) for j in file_paths[i] if j.count(".jpg") > 0] for i in file_paths}

            for i in file_paths: 
                for j in file_paths[i]:
                    image = cv2.imread(j)
                    prediction = model.predict(convertCV2toKeras(image, size_x=IMAGE_WIDTH, size_y=IMAGE_HEIGHT))
                    score = tf.nn.softmax(prediction)
                    classification = class_names[np.argmax(score)]
                    if classification != i: 
                        print(f"This image was misclassified as {classification} (should have been {i})")
                        cv2.imshow(str(j),image)
                        cv2.waitKey()
            if args.save: 
                if not os.path.exists(args.save): 
                    os.mkdir(args.save)
                model.save(args.save)
                print(f"Model saved to {args.save}")
                
if __name__ == "__main__": 
    main()