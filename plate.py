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
from utils import convertCV2toKeras
import tensorflow as tf
import numpy as np

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

    def link_model(self, model, key): 
        self.model = model
        self.key = key

    def annotate_images(self, model=None, key=None): 
        if not self.image_matrix: 
            raise LookupError(
                """
                Unable to find an image_matrix associated with this plate. 
                Please provide an image path on construction or use import_image and split_images()
                """)
        model = self.model if not model else model 
        if not model: 
            raise LookupError(
                """
                Unable to find an image model for predictions associated with this plate. 
                Please provide one or use link_model(). 
                """
            )
        key = self.key if not key else key
        if not key: 
            raise LookupError(
                """
                Unable to find an interpretation key to convert scores to label. Please provide one
                or use link_model(). 
                """
            )
        self.predictions_matrix = []
        self.score_matrix = []
        self.growth_matrix = []
        self.accuracy_matrix = []
        temp_score_row = []
        temp_predictions_row = []
        temp_growth_rows = []
        temp_accuracy_row = []
        for row in self.image_matrix: 
            for image in row: 
                prediction = self.model.predict(convertCV2toKeras(image)) 
                temp_predictions_row.append(prediction)
                score = tf.nn.softmax(prediction)
                temp_score_row.append(score)
                growth = self.key[np.argmax(score)]
                temp_growth_rows.append(growth)
                accuracy = np.max(score)
                temp_accuracy_row.append(accuracy)
            self.predictions_matrix.append(temp_predictions_row)
            self.score_matrix.append(temp_score_row)
            self.growth_matrix.append(temp_growth_rows)
            self.accuracy_matrix.append(temp_accuracy_row)
            temp_score_row = []
            temp_predictions_row = []
            temp_growth_rows = []
            temp_accuracy_row = []
        return self.growth_matrix