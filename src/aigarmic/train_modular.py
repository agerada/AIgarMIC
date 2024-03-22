#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	train_modular.py
# Author: 	Alessandro Gerada
# Date: 	2023-08-11
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Modular training function that works with binary or softmax"""

from aigarmic.train import visualise_training, train_binary, train_softmax
from aigarmic.file_handlers import save_training_log
from aigarmic.nn_design import (model_design_spectrum_2024_binary_first_step,
                                model_design_spectrum_2024_binary_second_step)
from aigarmic.file_handlers import predict_colony_images_from_directory

import os, cv2
import argparse
import pathlib
import warnings

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 160
BATCH_SIZE = 64


def main(): 
    parser = argparse.ArgumentParser("""
    This script loads images from annotations directory and trains ML to classify colony growth.
    """)
    parser.add_argument("annotations", type=str,
                        help="Directory containing annotated images")
    parser.add_argument("-v", "--visualise", action="store_true",
                        help="Generate visualisations for model diagnostics")
    parser.add_argument("-s", "--save", type=str,
                        help="If specified, tensorflow model will be saved to this folder")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Store performance log in output folder")
    parser.add_argument("-t", "--test_dataset", type=str,
                        help="Testing dataset for final model evaluation. Ideally unseen data."
                             "If not provided then input directory is used (whole dataset).")
    parser.add_argument("-m", "--model_type", type=str, choices=["binary", "softmax"],
                        default="binary", help="Model type to train [default] = binary")
    parser.add_argument("-c", "--complexity", type=int, default=0, choices=[0, 1],
                        help="Model complexity level [default] = 0. Supports 0 or 1. "
                             "0 = first-step model from Gerada et al Spectrum 2024;"
                             "1 = second-step model from Gerada et al Spectrum 2024.")
    args = parser.parse_args()

    annotated_images = pathlib.Path(args.annotations)
    print(f"Number of .jpg files{len(list(annotated_images.glob('*/*.jpg')))}")

    model_designs = (model_design_spectrum_2024_binary_first_step,
                     model_design_spectrum_2024_binary_second_step)

    if args.model_type == "binary":
        model, classes, history, results = train_binary(annotations_path=annotated_images,
                                                        model_design=model_designs[args.complexity](IMAGE_WIDTH,
                                                                                                    IMAGE_HEIGHT),
                                                        image_width=IMAGE_WIDTH,
                                                        image_height=IMAGE_HEIGHT,
                                                        batch_size=BATCH_SIZE)
    elif args.model_type == "softmax":
        model, classes, history, results = train_softmax(annotations_path=annotated_images,
                                                         model_design=model_designs[args.complexity](IMAGE_WIDTH,
                                                                                                     IMAGE_HEIGHT),
                                                         image_width=IMAGE_WIDTH,
                                                         image_height=IMAGE_HEIGHT,
                                                         batch_size=BATCH_SIZE)
    else:
        raise ValueError("Model type not supported")

    if args.save:
        if not os.path.exists(args.save): 
            os.mkdir(args.save)
        model.save(args.save)
        print(f"Model saved to {args.save}")
            
    if args.log: 
        if not args.save: 
            warnings.warn("Unable to save log file because model save path not provided, please use -s to provide path.")
        else:

            training_log_file = os.path.join(args.save, "training_log.csv")
            save_training_log(history, training_log_file)

    if args.test_dataset:
        if args.save is not None:
            annotation_log_file = os.path.join(args.save, "test_dataset_log.csv")
        else:
            annotation_log_file = None
        annotation_log = predict_colony_images_from_directory(args.test_dataset, model, classes, IMAGE_WIDTH, IMAGE_HEIGHT, args.model_type,
                                                              save_path=annotation_log_file)
        if args.visualise:
            for i in annotation_log:
                if i["predicted_class"] != i["true_class"]:
                    print(
                        f"This image was misclassified as {i['predicted_class']} "
                        f"with prediction of {i['prediction']} "
                        f"(should have been {i['true_class']})")
                    cv2.imshow(str(i['path']), i['image'])
                    cv2.waitKey()
                else:
                    print(f"Correct classification with prediction {i['prediction']} (class {i['true_class']})")

    if args.visualise:
        visualise_training(history)


if __name__ == "__main__":
    main()
