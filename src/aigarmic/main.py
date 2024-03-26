#!/usr/bin/env python
# Filename: 	main.py
# Author: 	Alessandro Gerada
# Date: 	2023-01-27
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Script to process images"""
import pathlib
from process_plate_image import split_by_grid
from plate import plate_set_from_dir
import argparse
from img_utils import get_concentration_from_path, get_paths_from_directory
import csv
from model import SoftmaxModel, BinaryModel, BinaryNestedModel
import sys
import cv2

MODEL_IMAGE_X = 160
MODEL_IMAGE_Y = 160
SUPPORTED_MODEL_TYPES = ['softmax', 'binary']


def main():
    parser = argparse.ArgumentParser(description="Main script to interpret agar dilution MICs",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('directory', type=str, help="""
        Directory containing images to process, arranged in sub-folders by antibiotic name, e.g.,: \n
        \t directory/ \n
        \t \t antibiotic1_name/ \n
        \t \t \t 0.jpg \n
        \t \t \t 0.125.jpg \n
        \t \t \t 0.25.jpg \n
        """)
    parser.add_argument("-m", "--model", type=str, nargs="*",
                        help="Specify one or more directories containing tensorflow model/s for image classification")
    parser.add_argument("-t", "--type_model", type=str, default="binary",
                        help="Type of keras model, e.g., binary, softmax [default] = binary")
    parser.add_argument("-o", "--output_file", type=str,
                        help="Specify output file for csv report (will be overwritten)")
    parser.add_argument("-v", "--validation_threshold", type=float, default=0.0,
                        help="Image annotation accuracies below this value will be validated for manual confirmation."
                             "0.0 checks no images [default]; 1.0 checks all images.")
    parser.add_argument("-c", "--check_contours", action="store_true", help="Check contours visually")
    parser.add_argument("-n", "--negative_codes", type=str,
                        help="Comma-separated list of no growth class codes for softmax model, e.g., 0,1 (default)")
    args = parser.parse_args()

    plate_images_paths = get_paths_from_directory(args.directory)

    if args.check_contours:
        cv2.startWindowThread()
        for abx, paths in plate_images_paths.items():
            for path in paths:
                _image = cv2.imread(path)
                try:
                    split_by_grid(_image, visualise_contours=True, plate_name=abx + '_' + str(get_concentration_from_path(path)))
                except ValueError as err:
                    print(err)

        pos_replies = ['y', 'yes', 'ye']
        neg_replies = ['n', 'no']
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        while True:
            input_key = input("""Completed contour checks. Would you like to continue with annotation? [Y / N]
                              Please only proceed if all images have correctly identified 96 boxes!
                              """)
            input_key = input_key.lower()
            if input_key in neg_replies:
                sys.exit()
            elif input_key in pos_replies:
                print("Continuing with annotation..")
                break
            else:
                print("Unable to recognise input, please try again..")
                continue

    if args.type_model not in SUPPORTED_MODEL_TYPES:
        sys.exit(f"Model type specified is not supported, please use one of {SUPPORTED_MODEL_TYPES}")

    if args.type_model == 'softmax' and len(args.model) != 1:
        sys.exit(
            """
            Softmax model can only run with one keras model
            """
        )

    if args.type_model == 'softmax':
        # Since args.model is a list, un-list
        [path_to_model] = args.model
        model = SoftmaxModel(path_to_model, trained_x=MODEL_IMAGE_X, trained_y=MODEL_IMAGE_Y)

    elif args.type_model == 'binary' and len(args.model) == 2:
        class_names_first_line = ['No growth', 'Growth']
        class_names_second_line = ['Poor growth', 'Good growth']
        first_line_model = BinaryModel(args.model[0], key=class_names_first_line, trained_x=MODEL_IMAGE_X,
                                       trained_y=MODEL_IMAGE_Y)
        second_line_model = BinaryModel(args.model[1], key=class_names_second_line, trained_x=MODEL_IMAGE_X,
                                        trained_y=MODEL_IMAGE_Y)
        model = BinaryNestedModel(first_line_model, second_line_model, first_model_accuracy_acceptance=0.6,
                                  suppress_first_model_accuracy_check=True)
    elif args.type_model == 'binary' and len(args.model) == 1:
        class_names = ['No growth', 'Growth']
        model = BinaryModel(args.model[0], key=class_names,
                            trained_x=MODEL_IMAGE_X, trained_y=MODEL_IMAGE_Y)
    else:
        sys.exit(f"Model type specified is not supported, please use one of {SUPPORTED_MODEL_TYPES}")

    abx_superset = {}
    parent_path = pathlib.Path(args.directory)
    for abx, paths in plate_images_paths.items():
        _plate_set = plate_set_from_dir(path=parent_path / abx,
                                        drug=abx,
                                        model=model)
        abx_superset[abx] = _plate_set

    for abx, plate_set in abx_superset.items():
        plate_set.review_poor_images(threshold=args.validation_threshold)
        if args.negative_codes:
            ng_codes = [int(x) for x in args.negative_codes.split(",")]
            plate_set.calculate_mic(no_growth_key_items=tuple(ng_codes))
        else:
            plate_set.calculate_mic()
        plate_set.generate_qc()

    if args.output_file:
        output_data = []
        for plate_set in abx_superset.values():
            output_data = output_data + plate_set.get_csv_data()
        with open(args.output_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, output_data[0].keys())
            writer.writeheader()
            writer.writerows(output_data)


if __name__ == "__main__":
    main()
