#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	draw_model.py
# Author: 	Alessandro Gerada
# Date: 	2023-08-08
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Draw neural network"""

from keras.models import Sequential
from keras import layers
from keras.utils import plot_model

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 160
initial_bias = None

growth_poor_growth = [

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

model = Sequential(growth_poor_growth)
model.save("8fsv.h5")
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
