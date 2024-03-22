# Filename: 	nn_design.py
# Author: 	Alessandro Gerada
# Date: 	2023-08-11
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Template NN designs as reported in Gerada et al. 2024 Microbiology Spectrum paper"""

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def model_design_spectrum_2024_binary_first_step(image_width: int,
                                                 image_height: int,
                                                 augmentation: bool = True):
    design = [
        #  spectrum 2024 first-step model
        layers.Rescaling(1./255, input_shape=(image_width, image_height, 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(32, (3,3), kernel_initializer='he_uniform', activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), kernel_initializer='he_uniform', activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1)
    ]

    data_augmentation = [
        layers.RandomFlip("horizontal",
                          input_shape=(image_width,
                                       image_height,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1)
    ]
    if augmentation:
        design = data_augmentation + design
    print(design)
    return Sequential(design)


def model_design_spectrum_2024_binary_second_step(image_width: int,
                                                  image_height: int,
                                                  augmentation: bool = True):
    design = [
        #  spectrum 2024 second-step model
        layers.Rescaling(1. / 255, input_shape=(image_width, image_height, 3)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1)
    ]

    data_augmentation = [
        layers.RandomFlip("horizontal",
                          input_shape=(image_width,
                                       image_height,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1)
    ]
    if augmentation:
        design = data_augmentation + design

    return Sequential(design)
