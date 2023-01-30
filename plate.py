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
from random import choice, randrange

@dataclass(order=True)
class Plate: 
    drug: str
    concentration: float
    image_path: Optional[str] = None
    nrow: int = 8
    ncol: int = 12

    def __post_init__(self): 
        type_check_fields = ["concentration"]
        for (name, field_type) in self.__annotations__.items(): 
            if name in type_check_fields and not isinstance(self.__dict__[name], field_type):
                current_type = type(self.__dict__[name])
                raise TypeError(f"The field `{name}` was assigned with `{current_type}` instead of `{field_type}`")
        if self.image_path: 
            self.image = cv2.imread(self.image_path)
            self.image_matrix = split_by_grid(self.image, self.nrow)

    def split_images(self): 
        self.image_matrix = split_by_grid(self.image, self.nrow)

    def import_image(self, image): 
        self.image = image

    def get_random_colony_image(self):
        i = randrange(self.nrow)
        j = randrange(self.ncol)
        code = self.drug + str(self.concentration) + "_i_" + str(i) + "_j_" + str(j)
        return self.image_matrix[i][j],code