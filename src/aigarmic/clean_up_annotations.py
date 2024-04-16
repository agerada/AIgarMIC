#!/usr/bin/env python
# Filename: 	clean_up_annotations.py
# Author: 	Alessandro Gerada
# Date: 	2023-03-17
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Script that cleans up duplicate images in annotation folders. Use clean_up_annotations.py -h for help."""

import argparse
import cv2  # pylint: disable=import-error
from aigarmic._img_utils import get_image_paths, Deleter, in_list


def clean_up_annotations_parser():
    parser = argparse.ArgumentParser(description="""
        Clean up duplicate images in annotation folders
        """)
    parser.add_argument('input_dir', type=str,
                        help="Input directory - can contain subdirectories of images which will be processed separately")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="Suppress file deletion warnings (CAUTION)")
    return parser


def main():
    parser = clean_up_annotations_parser()
    args = parser.parse_args()

    images_paths = get_image_paths(args.input_dir)
    
    deleter = Deleter(confirm=False if args.quiet else True)

    _images_list = []
    if isinstance(images_paths, list):
        for i in images_paths: 
            _image = cv2.imread(i)  # pylint: disable=no-member
            if not in_list(_image, _images_list): 
                _images_list.append(_image)
            else: 
                deleter.delete_file(i)

    if isinstance(images_paths, dict):
        for _, v in images_paths.items():
            _images_list = []
            for i in v: 
                _image = cv2.imread(i)  # pylint: disable=no-member
                if not in_list(_image, _images_list): 
                    _images_list.append(_image)
                else: 
                    deleter.delete_file(i)


if __name__ == "__main__":
    main()
