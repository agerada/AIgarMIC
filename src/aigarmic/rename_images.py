#!/usr/bin/env python
# Filename: 	rename_images.py
# Author: 	Alessandro Gerada
# Date: 	2023-03-23
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Script to batch the renaming of a set of agar dilution images. Use rename_images.py -h for help."""

import argparse
import os

from aigarmic._img_utils import rename_jpg


def main():
    parser = argparse.ArgumentParser(
        description='Script to batch rename images with halving value of drug concentration.')
    parser.add_argument('directory', type=str, 
                        help="""
                        Directory containing images. Filenames should reflect the concentration order when sorted, e.g.:
                        image1 (no drug)
                        image2 (low concentration)
                        image3 (medium concentration)
                        image4 (high concentration)
                        etc..
                        """)
    parser.add_argument('starting_concentration', type=float, 
                        help='Starting (max) concentration that will be iteratively halved')
    args = parser.parse_args()

    files = os.listdir(args.directory)
    files = [i for i in files if i.lower().count(".jpg") > 0]
    
    if not files: 
        raise ValueError("No files found.")

    files.sort()
    control_plate = files.pop(0)
    rename_jpg(os.path.join(args.directory, control_plate),
               os.path.join(args.directory, '0.jpg'))

    concentration = args.starting_concentration

    while files: 
        _file = files.pop()
        rename_jpg(os.path.join(args.directory, _file),
                   os.path.join(args.directory, f"{concentration}.jpg"))
        concentration /= 2

if __name__ == '__main__':
    main()
