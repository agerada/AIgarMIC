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
        print(f"linking model to plate {self.concentration}")
        self.model = model
        self.key = key

    def annotate_images(self, model=None, key=None): 
        print(f"annotating plate images - {self.concentration}")
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

    def print_matrix(self): 
        if not self.growth_matrix: 
            print(f"Plate {self.drug} - {self.concentration} not annotated")
        else: 
            for row in self.growth_matrix: 
                for result in row: 
                    if result == "Good growth": 
                        print("[+]", sep="", end="")
                        print(" ", end="", sep="")
                    else: 
                        print("[-]", sep="", end="")
                        print(" ", end="", sep="")
                print()

class PlateSet: 
    def __init__(self, plates_list) -> None:
        if not plates_list: 
            raise ValueError("Supply list of plates to create PlateSet")
        if any([not plate.growth_matrix for plate in plates_list]): 
            raise ValueError("Please run annotate_images() on plates before initialising PlateSet")

        _list_of_keys = [i.key for i in plates_list]
        if not all(i==_list_of_keys[0] for i in _list_of_keys): 
            raise ValueError("Plates supplied to PlateSet have different growth keys")
        if not _list_of_keys: 
            raise ValueError("Plates supplied to PlateSet do not have associated key")
        
        drug_names = [i.drug for i in plates_list]
        if len(set(drug_names)) > 1: 
            raise ValueError("Plates supplied to PlateSet have different antibiotic names")
        elif not len(set(drug_names)): 
            raise ValueError("Plates supplied to PlateSet do not have antibiotic names")
        else: 
            self.drug = plates_list[0].drug
            self.antibiotic_plates = [i for i in plates_list if i.concentration != 0.0]
            try: 
                _temp_positive_control_plate = [i for i in plates_list if i.concentration == 0.0]
                [self.positive_control_plate] = _temp_positive_control_plate
            except: 
                if not _temp_positive_control_plate: 
                    print(f"*Warning* - no control plate supplied to {self.drug} PlateSet")
                else: 
                    print(f"*Warning* - multiple control plates supplied to {self.drug} PlateSet, control plates will be skipped.")
            self.antibiotic_plates = sorted(self.antibiotic_plates)
            self.key = self.antibiotic_plates[0].key

        # check dimensions of plates' matrices
        if not self.valid_dimensions(): 
            raise ValueError("Plate matrices have different dimensions - unable to calculate MIC")

    def valid_dimensions(self): 
        matrices_array = [np.array(p.growth_matrix) for p in self.antibiotic_plates]
        matrices_shapes = [i.shape for i in matrices_array]
        return True if all(i==matrices_shapes[0] for i in matrices_shapes) else False

    def convert_mic_matrix(self, format = str): 
        allowed_formats = (str, float)
        if format not in allowed_formats: 
            raise ValueError(f"MIC matrix formats must be one of: {allowed_formats}")
        output = self.mic_matrix.astype(format)
        if format == str: 
            max_mic_plate = max([i.concentration for i in self.antibiotic_plates])
            min_mic_plate = min([i.concentration for i in self.antibiotic_plates])
            for i,row in enumerate(output): 
                for j,mic in enumerate(row): 
                    if float(mic) > max_mic_plate: 
                        output[i][j] = ">" + str(max_mic_plate)
                    elif float(mic) == min_mic_plate: 
                        output[i][j] = "<" + str(min_mic_plate)
                    else: 
                        output[i][j] = mic
        return output

    def calculate_MIC(self, format = str, no_growth_key_items = (0,1)): 
        _no_growth_names = [self.key[i] for i in no_growth_key_items]
        self.antibiotic_plates = sorted(self.antibiotic_plates, reverse=True)
        mic_matrix = np.array(self.antibiotic_plates[0].growth_matrix)
        mic_matrix = np.full(mic_matrix.shape, max([i.concentration for i in self.antibiotic_plates])*2)
        rows = range(mic_matrix.shape[0])
        cols = range(mic_matrix.shape[1])
        for plate in self.antibiotic_plates: 
            for row in rows: 
                for col in cols: 
                    if plate.growth_matrix[row][col] in _no_growth_names: 
                        mic_matrix[row][col] = plate.concentration
        self.mic_matrix = mic_matrix
        return mic_matrix

    def __repr__(self) -> str:
        return f"PlateSet of {self.drug} with {len(self.antibiotic_plates)} concentrations: {[i.concentration for i in self.antibiotic_plates]}"