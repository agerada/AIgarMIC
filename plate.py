#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	plate.py
# Author: 	Alessandro Gerada
# Date: 	2023-01-27
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Class implementation for plates"""

from process_plate_image import split_by_grid
from dataclasses import dataclass
from typing import Optional
import cv2

@dataclass
class Plate: 
    drug: str
    concentration: float
    image_path: Optional[str] = None
    nrow: int = 8
    ncol: int = 12

    def __post_init__(self): 
        if self.image_path: 
            print("ok")
            self.image = cv2.imread(self.image_path)
            self.image_matrix = split_by_grid(self.image, self.nrow)

    def split_images(self): 
        self.image_matrix = split_by_grid(self.image, self.nrow)

    def import_image(self, image): 
        self.image = image
