# Filename:      train.py
# Author:        Alessandro Gerada
# Date:          21/03/2024
# Copyright:     Alessandro Gerada 2024
# Email:         alessandro.gerada@liverpool.ac.uk

"""
Documentation
"""

import keras.callbacks

from file_handlers import create_dataset_from_directory
from utils import ValidationThresholdCallback
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.initializers import Constant
from sklearn.utils import class_weight
from tensorflow.keras import initializers

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def visualise_training(history: keras.callbacks.History):
    epochs = len(history.history['accuracy'])
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


def train_binary(annotations_path,
                 model_design: Sequential,
                 val_split: float = 0.2,
                 image_height: int = 160,
                 image_width: int = 160,
                 batch_size: int = 64,
                 epochs: int = 300,
                 stop_training_threshold: float = 0.98):
    label_mode = "binary"
    train_dataset, val_dataset = create_dataset_from_directory(
        path=annotations_path,
        label_mode=label_mode,
        image_width=image_width,
        image_height=image_height,
        val_split=val_split,
        batch_size=batch_size
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    if num_classes != 2:
        raise ValueError("Binary classification requires 2 classes")
    model_design = model_design

    full_dataset = image_dataset_from_directory(Path(annotations_path),
                                                image_size=(image_height, image_width),
                                                batch_size=batch_size,
                                                label_mode="binary")

    full_dataset_unbatched = tuple(full_dataset.unbatch())
    labels = [0, 0]
    for (_, label) in full_dataset_unbatched:
        labels[int(label.numpy())] += 1
    neg = labels[0]
    pos = labels[1]
    initial_bias = np.log([pos / neg])
    initial_bias = Constant(initial_bias)

    weight_0 = (1 / neg) * ((neg + pos) / 2)
    weight_1 = (1 / pos) * ((neg + pos) / 2)
    class_weights = {0: weight_0, 1: weight_1}

    model_design.add(layers.Dense(1, activation='sigmoid',
                                  bias_initializer=initial_bias))
    with tf.device('/device:GPU:0'):
        model_design.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=.0001),
                             loss='binary_crossentropy',
                             metrics=['accuracy'])

    val_callback = ValidationThresholdCallback(threshold=stop_training_threshold)
    history = model_design.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[val_callback]
    )
    results = model_design.evaluate(train_dataset, batch_size=batch_size, verbose=0)

    return model_design, class_names, history, results

def train_softmax(annotations_path,
                 model_design: Sequential,
                 val_split: float = 0.2,
                 image_height: int = 160,
                 image_width: int = 160,
                 batch_size: int = 64,
                 epochs: int = 300,
                 stop_training_threshold: float = 0.98):
    label_mode = "int"
    train_dataset, val_dataset = create_dataset_from_directory(
        path=annotations_path,
        label_mode=label_mode,
        image_width=image_width,
        image_height=image_height,
        val_split=val_split,
        batch_size=batch_size
    )
    class_names = train_dataset.class_names
    num_classes = len(class_names)
    obs_dict = dict(zip(list(range(num_classes)), [0] * num_classes))
    all_class_obs = []
    for i, j in train_dataset:
        for k in j:
            k_arr = k.numpy()
            obs_dict[k_arr] += 1
            all_class_obs.append(k_arr)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(all_class_obs), y=all_class_obs)
    class_weights = dict(zip(list(range(num_classes)), class_weights))
    initial_bias = initializers.Zeros()
    # class_weights = None
    # initial_bias = None

    model_design.add(layers.Dense(num_classes, activation='softmax', bias_initializer=initial_bias))

    with tf.device('/device:GPU:0'):
        model_design.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        val_callback = ValidationThresholdCallback(threshold=stop_training_threshold)
        history = model_design.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=[val_callback]
        )

    results = model_design.evaluate(train_dataset, batch_size=batch_size, verbose=0)
    return model_design, class_names, history, results


