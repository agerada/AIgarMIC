# -*- coding: utf-8 -*-
# Filename: 	file_handlers.py
# Author: 	Alessandro Gerada
# Date: 	2023-08-11
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Functions to facilitate working with image data files"""
import pathlib

from aigarmic.utils import convertCV2toKeras
import csv
import os
from os import path
from typing import Optional, Union
import cv2
import keras.callbacks
import numpy as np
import tensorflow as tf


def create_dataset_from_directory(directory: str,
                                  label_mode: str,
                                  image_width: int,
                                  image_height: int,
                                  seed: int = 12345,
                                  val_split: float = 0.2,
                                  batch_size: int = 32) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create a training and validation dataset from a directory containing subdirectories for each class of image data.

    :param directory: path containing images, each in subdirectory corresponding to class
    :param label_mode: Labelling depending on model type ("binary" for binary or "int" for softmax)
    :param image_width: Image width in pixels
    :param image_height: Image height in pixels
    :param seed: Random seed for dataset splitting
    :param val_split: Proportion of data to use for validation
    :param batch_size: Batch size for datasets
    :return: Tuple containing training and validation datasets
    """
    train_dataset = tf.keras.utils.image_dataset_from_directory(
                directory,
                validation_split=val_split, 
                subset='training',
                seed=seed,
                image_size=(image_width, image_height), 
                batch_size=batch_size, 
                label_mode=label_mode
            )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=val_split, 
        subset='validation', 
        seed=seed,
        image_size=(image_width, image_height), 
        batch_size=batch_size, 
        label_mode=label_mode
    )
    print(f"Found the following labels/classes: {train_dataset.class_names}")
    return train_dataset, val_dataset


def predict_colony_images_from_directory(directory: Optional[Union[str, pathlib.Path]],
                                         model: tf.keras.models.Model,
                                         class_names: list[str],
                                         image_width: int,
                                         image_height: int,
                                         model_type: str,
                                         save_path: Optional[Union[str, pathlib.Path]] = None,
                                         binary_threshold: float = 0.5) -> list[dict]:
    """
    Predict the class of images in a directory using a trained model, and compare prediction to
    true class (based on subdirectory in which image is located, which should correspond to class name).

    :param directory: Directory containing images to predict
    :param model: Model to use for prediction
    :param class_names: List of class names
    :param image_width: Image width in pixels
    :param image_height: Image height in pixels
    :param model_type: "binary" or "softmax"
    :param save_path: Path to save prediction log
    :param binary_threshold: For binary models, threshold for classifying as positive
    :return: List of dictionaries containing image, path, prediction, predicted class, and true class (for each image)
    """
    output = []
    file_paths = {i: os.listdir(os.path.join(directory, i)) for i in class_names}
    # add subdirectories
    file_paths = {i: [os.path.join(directory, i, j) for j in file_paths[i] if j.count(".jpg") > 0] for i in file_paths}

    for i in file_paths:
        for j in file_paths[i]:
            image = cv2.imread(j)
            true_class = i
            directory = j
            prediction = model.predict(convertCV2toKeras(image, image_width, image_height))
            if model_type == "binary":
                [prediction] = prediction.reshape(-1)
                predicted_class = class_names[0] if prediction <= binary_threshold else class_names[1]
            elif model_type == "softmax":
                prediction = tf.nn.softmax(prediction)
                prediction_value = np.max(prediction)
                predicted_class = np.argmax(prediction)
                prediction = prediction_value
            else:
                raise ValueError("Model type not supported")
            output.append({"image": image,
                           "path": directory,
                           "prediction": prediction,
                           "predicted_class": predicted_class,
                           "true_class": true_class})
    if save_path:
        with open(save_path, "w") as file:
            writer = csv.DictWriter(file,
                                    fieldnames=['path', 'prediction', 'predicted_class', 'true_class'],
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(output)
    return output


def save_training_log(model_history: keras.callbacks.History,
                      save_path: Union[str, pathlib.Path]) -> None:
    """
    Save training log to CSV file

    :param model_history: Training history object
    :param save_path: Directory to save training log
    """
    with open(save_path, "w") as file:
        writer = csv.writer(file)
        h = zip(
            model_history.history['accuracy'],
            model_history.history['val_accuracy'],
            model_history.history['loss'],
            model_history.history['val_loss'],
            range(len(model_history.history['accuracy'])))
        writer.writerow(['accuracy', 'val_accuracy', 'loss', 'val_loss', 'epoch'])
        writer.writerows(h)
