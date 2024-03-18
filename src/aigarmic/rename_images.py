#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	rename_images.py
# Author: 	Alessandro Gerada
# Date: 	2023-03-23
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Script to batch the renaming of images"""

import argparse
import os

def rename_jpg(file_in, file_out): 
    if not file_in or not file_out: 
        return
    print(f"Renaming {file_in} to {file_out}")
    os.rename(file_in, file_out)

def main(): 
    parser = argparse.ArgumentParser(description='Script to batch rename images with halving value of drug concentration.')
    parser.add_argument('directory', type=str, 
                        help="""
                        Directory containing images. Images names should reflect the concentration order when sorted, e.g.:
                        image1 (no drug)
                        image2 (low concentration)
                        image3 (medium concentration)
                        image4 (high concentration) 
                        """)
    parser.add_argument('starting_concentration', type=float, 
                        help='Starting concentration that will be iteratively halved')
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

main()
