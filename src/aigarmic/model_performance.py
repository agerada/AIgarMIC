#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	model_performance.py
# Author: 	Alessandro Gerada
# Date: 	2023-03-16
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Test the performance of a saved model on a set of colony images. Use model_performance.py -h for help."""

from aigarmic.file_handlers import get_paths_from_directory
from aigarmic.model import BinaryModel
import cv2  # pylint: disable=import-error
import argparse


def model_performance_parser():
    parser = argparse.ArgumentParser(description="Evaluate saved model performance")
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("annotations_path", type=str, help="Path to the annotations to test")
    parser.add_argument("-x", "--trained_x", type=int, default=160,
                        help="Width of the images that the model was trained on [default=160]")
    parser.add_argument("-y", "--trained_y", type=int, default=160,
                        help="Height of the images that the model was trained on [default=160]")
    parser.add_argument("-t", "--threshold", type=float, default=0.95,
                        help="Accuracy threshold, to highlight low accuracy predictions [default=0.95]")
    return parser


def main():
    parser = model_performance_parser()
    args = parser.parse_args()

    images = get_paths_from_directory(args.annotations_path)
    model = BinaryModel(args.model_path,
                        trained_x=args.trained_x,
                        trained_y=args.trained_y,
                        key=['No growth', 'Growth'])

    results = []
    for i, paths in images.items():
        for path in paths:
            image = cv2.imread(path)  # pylint: disable=no-member
            p = model.predict(image)
            p['true_class'] = int(i)
            results.append(p)

    color_end = '\033[0m'
    color_red = '\033[91m'
    color_green = '\033[92m'

    errors = 0
    for i in results:
        if i['growth_code'] != i['true_class']:
            errors += 1
            color = color_red
        else:
            color = color_green
        print(f"Prediction={i['prediction']:.2f} \t "
              f"Growth code={color}{i['growth_code']}{color_end} \t "
              f"True class={color}{i['true_class']}{color_end} \t "
              f"Accuracy={color_red if i['accuracy'] < args.threshold else color_green}{i['accuracy']:.2f}{color_end}")

    print()
    print(f"Errors: {errors} from {len(results)} images ({errors/len(results)*100:.2f}%)")

    threshold = 0.9
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    n = len(results)
    for i in results:
        if i['growth_code'] != i['true_class']:
            if i['accuracy'] < threshold:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if i['accuracy'] < threshold:
                false_positives += 1
            else:
                true_negatives += 1

    print(f"TP={true_positives} ({true_positives/n*100:.2f}%)")
    print(f"FP={false_positives} ({false_positives/n*100:.2f}%)")
    print(f"TN={true_negatives} ({true_negatives/n*100:.2f}%)")
    print(f"FN={false_negatives} ({false_negatives/n*100:.2f}%)")


if __name__ == "__main__":
    main()
