# Filename:      train.py
# Author:        Alessandro Gerada
# Date:          21/03/2024
# Copyright:     Alessandro Gerada 2024
# Email:         alessandro.gerada@liverpool.ac.uk

"""
Functions and classes that allow for training neural network models for colony image classification.
"""

from aigarmic.file_handlers import create_dataset_from_directory
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.initializers import Constant
import keras.callbacks
from sklearn.utils import class_weight
from tensorflow.keras import initializers
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def train_binary(annotations_path,
                 model_design: Sequential,
                 val_split: float = 0.2,
                 image_width: int = 160,
                 image_height: int = 160,
                 batch_size: int = 64,
                 epochs: int = 300,
                 stop_training_threshold: float = 0.98,
                 learning_rate: float = .0001) -> tuple[Sequential, list[str], keras.callbacks.History, list]:
    """
    Train a binary classification model to differentiate between two classes of colony images.
    Provide a keras sequential model design to inform the neural network architecture. The final binary/sigmoid layer
    should not be included in the sequential model design, as this is added by the function.

    :param annotations_path: Path to directory containing annotated images, with subdirectories for each class (usually
        '0' and '1')
    :param model_design: Keras model design (Sequential) to inform neural network architecture, excluding final layer
    :param val_split: Validation split proportion
    :param image_width: Image width (pixels)
    :param image_height: Image height (pixels)
    :param batch_size: Training batch size
    :param epochs: Max number of training epochs
    :param stop_training_threshold: Accuracy threshold at which to accept model and stop training
    :param learning_rate: Learning rate for the optimizer

    :return: Tuple containing trained model, class names, training history, and evaluation results
    """
    label_mode = "binary"
    train_dataset, val_dataset = create_dataset_from_directory(
        directory=annotations_path,
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
        model_design.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
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
                  stop_training_threshold: float = 0.98) -> tuple[Sequential, list[str], keras.callbacks.History, list]:
    """
    Train a softmax classification model to differentiate between multiple classes (2 or more) of colony images.
    The final softmax layer should not be included in the sequential model design, as this is added by the function.

    :param annotations_path: Path to directory containing annotated images, with subdirectories for each class
        (e.g., '0', '1', '2', ...)
    :param model_design: Keras model design (Sequential) to inform neural network architecture, excluding final layer
    :param val_split: Validation split proportion
    :param image_width: Image width (pixels)
    :param image_height: Image height (pixels)
    :param batch_size: Training batch size
    :param epochs: Max number of training epochs
    :param stop_training_threshold: Accuracy threshold at which to accept model and stop training

    :return: Tuple containing trained model, class names, training history, and evaluation results
    """
    label_mode = "int"
    train_dataset, val_dataset = create_dataset_from_directory(
        directory=annotations_path,
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
    for _, j in train_dataset:
        for k in j:
            k_arr = k.numpy()
            obs_dict[k_arr] += 1
            all_class_obs.append(k_arr)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(all_class_obs),
                                                      y=all_class_obs)
    class_weights = dict(zip(list(range(num_classes)), class_weights))
    initial_bias = initializers.Zeros()

    model_design.add(layers.Dense(num_classes, activation='softmax', bias_initializer=initial_bias))

    with tf.device('/device:GPU:0'):
        model_design.compile(optimizer='adam',
                             loss='sparse_categorical_crossentropy',
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


def visualise_training(history: keras.callbacks.History) -> None:
    """
    Visualise training and validation accuracy and loss over epochs.

    :param history: Training history object (usually returned by model.fit(), train_binary(), or train_softmax())
    """
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


class ValidationThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        """
        Stop training if validation accuracy is above threshold

        :param threshold: Threshold to stop training
        """
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None) -> None:
        """
        Determines whether to stop training based on validation accuracy
        """
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True
