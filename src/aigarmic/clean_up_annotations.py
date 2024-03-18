#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	clean_up_annotations.py
# Author: 	Alessandro Gerada
# Date: 	2023-03-17
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Script that cleans up duplicate images in annotation folders"""

import argparse
import cv2
from utils import get_image_paths
import numpy as np
from os import remove

class Deleter: 
    def __init__(self, confirm = True): 
        self.confirm = confirm
        self.converter = {
            "y": "yes", 
            "ye": "yes", 
            "yes": "yes", 
            "n": "no", 
            "no": "no", 
            "a": "all", 
            "al": "all", 
            "all": "all"
        }
    def delete_file(self, file): 
        if not self.confirm: 
            print(f"Deleting duplicate file: {file}")
            remove(file)
        else: 
            while True: 
                user_in = input(f"[ Yes / No / All ] Delete duplicate file: {file}").lower()
                if user_in not in self.converter: 
                    print("Input not recognised, please repeat")
                    continue
                if self.converter[user_in] == "yes": 
                    print(f"Deleting file: {file}")
                    remove(file)
                    break
                if self.converter[user_in] == "no": 
                    print(f"Skipping file: {file}")
                    break
                if self.converter[user_in] == "all": 
                    print("Warnings suppressed")
                    print(f"Deleting file: {file}")
                    remove(file)
                    self.confirm = False
                    break
                else: 
                    raise LookupError("Unable to interpret user loop")


def is_similar(image1, image2):
    """
    From: https://stackoverflow.com/a/23199159
    """
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())

def in_list(image, list): 
    if not list: 
        return False
    else: 
        for i in list: 
            if is_similar(i, image): 
                return True
        return False

def main(): 
    parser = argparse.ArgumentParser("""
    Clean up duplicate images in annotation folders
    """)
    parser.add_argument('input_dir', type=str, help='Input directory - can contain subdirectory of images which will be processed separately')
    parser.add_argument('-q', '--quiet', action='store_true', help="Suppress file deletion warnings (CAUTION)")
    args = parser.parse_args()

    images_paths = get_image_paths(args.input_dir)
    
    deleter = Deleter(confirm= False if args.quiet else True)

    _images_list = []
    if type(images_paths) == list: 
        for i in images_paths: 
            _image = cv2.imread(i)
            if not in_list(_image, _images_list): 
                _images_list.append(_image)
            else: 
                deleter.delete_file(i)

    if type(images_paths) == dict: 
        for k,v in images_paths.items(): 
            _images_list = []
            for i in v: 
                _image = cv2.imread(i)
                if not in_list(_image, _images_list): 
                    _images_list.append(_image)
                else: 
                    deleter.delete_file(i)

main()