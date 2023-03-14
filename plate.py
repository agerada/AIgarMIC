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
import os 
from string import ascii_uppercase

@dataclass(order=True)
class Plate: 
    drug: str
    concentration: float
    image_path: Optional[str] = None
    nrow: int = 8
    ncol: int = 12
    visualise_contours: bool = False

    def __post_init__(self): 
        type_check_fields = ["concentration"]
        for (name, field_type) in self.__annotations__.items(): 
            if name in type_check_fields and not isinstance(self.__dict__[name], field_type):
                current_type = type(self.__dict__[name])
                raise TypeError(f"The field `{name}` was assigned with `{current_type}` instead of `{field_type}`")
        if self.image_path: 
            self.image = cv2.imread(self.image_path)
            self.image_matrix = split_by_grid(self.image, self.nrow, visualise_contours=self.visualise_contours, 
                                              plate_name=self.drug + '_' + str(self.concentration))

    def split_images(self): 
        self.image_matrix = split_by_grid(self.image, self.nrow, 
                                          visualise_contours=self.visualise_contours, 
                                          plate_name=self.drug + '_' + str(self.concentration))

    def import_image(self, image): 
        self.image = image

    def get_colony_image(self, index = None):
        """
        Pulls colony image and associated codestamp
        If no index is provided (default) a random image is given
        """
        if index: 
            try: 
                i,j = index
                image = self.image_matrix[i][j]
            except Exception as e: 
                print(f"Invalid index provided to get_colony_image: {index}")                 
                print(e)
        else: 
            i = randrange(self.nrow)
            j = randrange(self.ncol)
            image = self.image_matrix[i][j]
        code = self.drug + str(self.concentration) + "_i_" + str(i) + "_j_" + str(j)
        return image,code

    def link_model(self, model, key, model_image_x = 160, model_image_y = 160): 
        """
        model_image_x and model_image_y are the pixel dimensions used to train the model
        """
        print(f"linking model to plate {self.concentration}")
        self.model = model
        self.key = key
        self.model_image_x = model_image_x
        self.model_image_y = model_image_y

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
        self.growth_code_matrix = []
        self.accuracy_matrix = []
        temp_score_row = []
        temp_predictions_row = []
        temp_growth_rows = []
        temp_growth_code_rows = []
        temp_accuracy_row = []
        for row in self.image_matrix: 
            for image in row: 
                prediction = self.model.predict(convertCV2toKeras(image, size_x=self.model_image_x, size_y=self.model_image_y)) 
                temp_predictions_row.append(prediction)
                score = tf.nn.softmax(prediction)
                temp_score_row.append(score)
                growth_code = np.argmax(score)
                growth = self.key[growth_code]
                temp_growth_rows.append(growth)
                temp_growth_code_rows.append(growth_code)
                accuracy = np.max(score)
                temp_accuracy_row.append(accuracy)
            self.predictions_matrix.append(temp_predictions_row)
            self.score_matrix.append(temp_score_row)
            self.growth_matrix.append(temp_growth_rows)
            self.growth_code_matrix.append(temp_growth_code_rows)
            self.accuracy_matrix.append(temp_accuracy_row)
            temp_score_row = []
            temp_predictions_row = []
            temp_growth_rows = []
            temp_accuracy_row = []
            temp_growth_code_rows = []
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

    def get_inaccurate_images(self, threshold = .9): 
        output = set()
        for i,row in enumerate(self.accuracy_matrix): 
            for j,item in enumerate(row): 
                if item < threshold: 
                    output.add((i,j))
        return output
    
    def review_poor_images(self, threshold = .9, save_dir = None): 
        codes = {48: 0, 49: 1, 50: 2, 27: "esc", 13: "enter"}
        inaccurate_images_indexes = self.get_inaccurate_images(threshold) 
        changed_log = []
        for image_index in inaccurate_images_indexes: 
            image,stamp = self.get_colony_image(image_index)
            i,j = image_index
            growth = self.growth_matrix[i][j]
            accuracy = self.accuracy_matrix[i][j]
            print()
            print(f"This image was labelled as {growth} with an accuracy of {accuracy * 100:.2f}")
            cv2.imshow(str(self.concentration) + f" position {i} {j}", image)
            print("Press enter to continue, or enter new classification: ")
            while True: 
                input_key = cv2.waitKey()
                if input_key not in codes: 
                    print("Input not recognised, please try again..")
                    continue
                else: 
                    break
            input_code = codes[input_key]
            if input_code == "esc" or input_code == "enter" or self.key[input_code] == growth: 
                print("Classification not changed.")
                continue
            else: 
                # reassign growth
                print(f"Reassigning image to {self.key[input_code]}")
                self.growth_matrix[i][j] = self.key[input_code]
                self.growth_code_matrix[i][j] = input_code
                changed_log.append(image_index)
                if save_dir: 
                    if not os.path.exists(save_dir):
                        print(f"Creating directory: {save_dir}")
                        os.mkdir(save_dir)
                    class_dir = os.path.join(save_dir, str(input_code))
                    if not os.path.exists(class_dir): 
                        print(f"Creating class subdirectory: {class_dir}")
                        os.mkdir(class_dir)
                    save_path = os.path.join(class_dir, stamp + ".jpg")
                    print(f"Saving image to: {save_path}")
                    cv2.imwrite(save_path, image)
        return changed_log # return list of changed indexes
        
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

    def get_all_plates(self): 
        return sorted(self.antibiotic_plates + [self.positive_control_plate])
    
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
    
    def generate_QC(self): 
        qc_matrix = np.full(self.mic_matrix.shape, fill_value="", dtype=str)

        try:
            for i,row in enumerate(self.positive_control_plate.growth_matrix): 
                for j,item in enumerate(row): 
                    if item != "Good growth": 
                        qc_matrix[i][j] = "F"
                    else: 
                        qc_matrix[i][j] = "P"
        except: 
            print(f"*Warning* - {repr(self)} does not contain a positive control plate.")
            print("QC not valid.")
        
        def remove_ones(code): 
            return 0 if code == 1 else code
        
        antibiotic_plates = sorted(self.antibiotic_plates, reverse=True)
        if len(antibiotic_plates) > 1: 
            rows = range(qc_matrix.shape[0])
            cols = range(qc_matrix.shape[1])
            for i in rows: 
                for j in cols: 
                    previous_growth_code = remove_ones(antibiotic_plates[0].growth_code_matrix[i][j])
                    flipped = False # we only allow one "flip" from no growth -> growth
                    for k in antibiotic_plates[1:]: 
                        next_growth_code = remove_ones(k.growth_code_matrix[i][j])
                        if next_growth_code < previous_growth_code: 
                            qc_matrix[i][j] = "W"
                        if next_growth_code != previous_growth_code: 
                            if not flipped: 
                                flipped = True
                            else: 
                                qc_matrix[i][j] = "W"
                        previous_growth_code = next_growth_code
        else: 
            print(f"*Warning* - {repr(self)} has insufficient plates for QC")
        self.qc_matrix = qc_matrix
        return qc_matrix

    def review_poor_images(self, threshold = .9, save_dir = None): 
        changed = [i.review_poor_images(threshold, save_dir) for i in self.get_all_plates()]
        print(f"{len(changed)} images re-classified.")

    def get_csv_data(self, format="l"): 
        mic_matrix_str = self.convert_mic_matrix(str)
        if format == 'l': 
            row_letters = ascii_uppercase[0:len(mic_matrix_str)]
            col_nums = [i + 1 for i in range(len(mic_matrix_str[0]))]
            output = []
            for i in range(len(row_letters)): 
                for j in range(len(col_nums)): 
                    position = row_letters[i]+str(col_nums[j])
                    mic = mic_matrix_str[i][j]
                    qc = self.qc_matrix[i][j]
                    output.append({'Antibiotic': self.drug, 'Position': position, 'MIC': mic, 'QC': qc})
            return output
        
    def __repr__(self) -> str:
        return f"PlateSet of {self.drug} with {len(self.antibiotic_plates)} concentrations: {[i.concentration for i in self.antibiotic_plates]}"