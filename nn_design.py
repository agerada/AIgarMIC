#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	nn_design.py
# Author: 	Alessandro Gerada
# Date: 	2023-08-11
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Sequential NN designs"""
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from train_modular import IMAGE_WIDTH,IMAGE_HEIGHT

data_augmentation = Sequential([
    layers.RandomFlip("horizontal",
                    input_shape=(IMAGE_WIDTH,
                                IMAGE_WIDTH,
                                3)),
    layers.RandomRotation(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1)
])

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
        layers.Dropout(0.1), 
        #layers.Dense(1, activation='sigmoid', bias_initializer = initial_bias)
    ]

growth_poor_growth = [ 
        
        # Working model for second line
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'), 
        layers.Dropout(0.1), 
        layers.Dense(128, activation='relu'), 
        layers.Dropout(0.1), 
        #layers.Dense(1, activation='sigmoid', bias_initializer = initial_bias)
    ]

simple = [
    layers.Rescaling(1./255, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
        layers.Conv2D(8, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(8, activation='relu'), 
        layers.Dropout(0.1)
]

keras_model = [
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
    layers.Cropping2D(10),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
]

growth_no_growth = [ 
        # Current working model for first line

        data_augmentation, 
        layers.Rescaling(1./255, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
        layers.Cropping2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, kernel_initializer='he_uniform', activation='relu'), 
        layers.MaxPooling2D(), 

        layers.Conv2D(64, 3, kernel_initializer='he_uniform', activation='relu'), 
        layers.MaxPooling2D(), 

        layers.Flatten(),
        layers.Dense(64, activation='relu')
        #layers.Dropout(0.5), 
        #layers.Dense(1, activation='sigmoid', bias_initializer = initial_bias)
    ]