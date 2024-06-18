# Filename: 	plate.py
# Author: 	Alessandro Gerada
# Date: 	2023-01-27
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Class implementation for plates"""
from pathlib import Path
from aigarmic.process_plate_image import split_by_grid
from aigarmic.model import Model
from aigarmic._img_utils import get_image_paths
from aigarmic.file_handlers import get_concentration_from_path
from typing import Optional, Union
import cv2  # pylint: disable=import-error
from random import randrange
import numpy as np
import os
from string import ascii_uppercase
from warnings import warn


class Plate:
    def __init__(self, drug: str,
                 concentration: float,
                 image: Optional[Union[str, cv2.typing.MatLike]] = None,
                 n_row: Optional[int] = None,
                 n_col: Optional[int] = None,
                 growth_code_matrix: Optional[list[list[int]]] = None,
                 visualise_contours: bool = False,
                 model: Optional[Model] = None,
                 key: Optional[list[str]] = None) -> None:
        """
        Store and process an agar plate image

        :param drug: Antibiotic name
        :param concentration: Antibiotic concentration
        :param image: CV2 image array or the path to the image
        :param n_row: Number of rows in the plate
        :param n_col: Number of columns in the plate
        :param visualise_contours: Visualise the contours of the plate (useful for validation of grid splitting)
        :param model: Model to use for predictions
        :param key: Key to interpret model output (try to infer from model if not provided)
        """
        self.drug = drug
        self.concentration = concentration
        self.image = image
        self.n_row = n_row
        self.n_col = n_col
        self.accuracy_matrix = None
        self.model = model
        self.key = None
        if key is not None:
            if self.model is not None and self.model.get_key() != key:
                warn(f"Key provided to Plate does not match linked model key: {key} vs {self.model.get_key()}")
                warn(f"Plate will be using key parameter: {key}")
            self.key = key
        else:
            if self.model is not None:
                try:
                    self.key = self.model.get_key()
                except LookupError:
                    warn(f"No key found for linked model: {self.model}")

        if growth_code_matrix is not None:
            self.add_growth_code_matrix(growth_code_matrix)
        else:
            self.growth_code_matrix = None
        self.growth_matrix = None
        self.score_matrix = None
        self.predictions_matrix = None
        self.image_matrix = None
        self.model_image_x = None
        self.model_image_y = None

        if self.image is not None:
            if n_row is None or n_col is None:
                raise ValueError("Plate dimensions must be provided if image path is provided")

            if isinstance(self.image, str):
                self.image = cv2.imread(self.image)  # pylint: disable=no-member

            self.image_matrix = split_by_grid(self.image, self.n_row, visualise_contours=visualise_contours,
                                              plate_name=self.drug + '_' + str(self.concentration))

            if self.growth_code_matrix is not None:
                warn("Growth code matrix provided along with image. Image will be saved but not used for annotations.")

            if self.model is not None:
                self.annotate_images()

    def add_growth_code_matrix(self, growth_code_matrix: list[list[int]]) -> None:
        if not self.valid_growth_code_matrix(growth_code_matrix):
            raise ValueError("Invalid growth code matrix, please provide a 2D growth coe matrix (list), values"
                             "must be integers, and cannot be negative")
        self.growth_code_matrix = growth_code_matrix
        dim_x, dim_y = self.matrix_dimensions(growth_code_matrix)
        if self.n_row is not None or self.n_col is not None:
            if dim_x != self.n_row or dim_y != self.n_col:
                raise ValueError(f"Dimensions of growth code matrix do not match plate dimensions: "
                                 f"{dim_x}x{dim_y} vs {self.n_row}x{self.n_col}")
        else:
            self.n_row = dim_x
            self.n_col = dim_y

    def valid_growth_code_matrix(self, growth_code_matrix: list[list[int]]) -> bool:
        try:
            self.matrix_dimensions(growth_code_matrix)
        except ValueError:
            return False

        for i in growth_code_matrix:
            for j in i:
                if not isinstance(j, int):
                    return False
                if self.key is not None:
                    if j > len(self.key) - 1:
                        return False
                if j < 0:
                    return False
        return True

    @staticmethod
    def matrix_dimensions(matrix) -> tuple[int, ...]:
        """
        Get dimensions of a matrix

        :param matrix: matrix to get dimensions of
        :raises ValueError: if matrix is not 2D or is not a valid matrix
        :return: tuple of dimensions
        """
        try:
            dimensions = np.shape(matrix)
            if len(dimensions) != 2:
                raise ValueError(f"Matrix {matrix} is not a 2D matrix")
            else:
                return dimensions
        except ValueError as e:
            raise ValueError(f"Matrix {matrix} is not a valid matrix") from e

    def split_images(self, visualise_contours: bool = False) -> None:
        """
        Splits images into individual colony images using grid

        :param visualise_contours: Visualise the contours of the plate (useful for validation of grid splitting)
        """
        self.image_matrix = split_by_grid(self.image, self.n_row,
                                          visualise_contours=visualise_contours,
                                          plate_name=self.drug + '_' + str(self.concentration))

    def import_image(self, image: np.ndarray) -> None:
        """
        Import and save image of agar plate

        :param image: loaded using cv2.imread
        """
        self.image = image

    def get_colony_image(self, index: Optional[tuple[int, int]] = None) -> tuple[np.ndarray, str]:
        """
        Pulls colony image and associated code-stamp
        Code-stamps are strings containing, in sequence:
        - Antibiotic name
        - Antibiotic concentration
        - Row (i) index
        - Column (j) index

        If no index is provided (default) a random image is given

        @param index: tuple of row and column index
        @return: tuple of image and code-stamp (e.g., "drug_0.125_i_1_j_2")
        """
        if index is None:
            i = randrange(self.n_row)
            j = randrange(self.n_col)
            image = self.image_matrix[i][j]
        else:
            try:
                i, j = index
                image = self.image_matrix[i][j]
            except KeyError as e:
                raise KeyError(f"Invalid index provided to get_colony_image: {index}") from e
        code = self.drug + "_" + str(self.concentration) + "_i_" + str(i) + "_j_" + str(j)
        return image, code

    def link_model(self, model: Model) -> None:
        """
        Link model to plate for predictions

        :param model: Model to link
        """
        self.model = model
        self.model_image_x = model.trained_x
        self.model_image_y = model.trained_y

    def get_key(self) -> Optional[list[str]]:
        """
        Get key from linked Model

        :raises: LookupError: No linked model to get key from
        :return: Key (or None if one is not found)
        """
        if self.key is not None:
            return self.key
        elif self.model is not None:
            return self.model.get_key()
        else:
            return None

    def set_key(self, key: list[str]) -> None:
        """
        Set plate key. Checks whether differs from linked model key (if any),
        and warns if different.

        :param key: List of growth categories (zero-indexed)
        """
        if self.model is not None:
            if self.model.get_key() != key:
                warn(f"Key provided to Plate does not match linked model key: {key} vs {self.model.get_key()}")
                warn(f"Plate will be overriding key parameter: {key}")
        self.key = key

    def annotate_images(self, model: Optional[Model] = None) -> list[list[str]]:
        """
        Annotate plate images

        :param model: linked model to use for predictions
        :return: Two-dimensional list of growth annotations
        """
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
                prediction_data = model.predict(image)
                if 'growth_code' not in prediction_data:
                    raise ValueError("Model predictions dict must contain 'growth_code'")
                temp_predictions_row.append(prediction_data.get('prediction', None))
                temp_score_row.append(prediction_data.get('score', None))
                temp_growth_code_rows.append(prediction_data['growth_code'])
                _growth = None
                if 'growth' not in prediction_data:
                    try:
                        key = model.get_key()
                        _growth = key[prediction_data['growth_code']]
                    except LookupError:
                        pass
                temp_growth_rows.append(_growth)
                temp_accuracy_row.append(prediction_data.get('accuracy', 1.))
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

    def print_matrix(self) -> None:
        """
        Print growth matrix in human-readable format
        """
        if not self.growth_matrix:
            print(f"Plate {self.drug} - {self.concentration} not annotated")
        else:
            for row in self.growth_matrix:
                for result in row:
                    print(str(result), sep="", end="")
                    print(" ", end="", sep="")
                print()

    def get_inaccurate_images(self, threshold: float = .9) -> set[tuple[int, int]]:
        """
        Get indexes of images with prediction accuracy below threshold
        :param threshold: Prediction threshold
        :return: Set containing indices of inaccurate images
        """
        output = set()
        for i, row in enumerate(self.accuracy_matrix):
            for j, item in enumerate(row):
                if item < threshold:
                    output.add((i, j))
        return output

    def review_poor_images(self, threshold: float = .9,
                           save_dir: str = None) -> list[tuple[int, int]]:
        """
        Review and re-classify images with prediction accuracy below threshold. Classes should be zero indexed (e.g.,
        0, 1, 2). Currently, only supports up to 9.
        If save_dir provided then colony images will also be saved to a subdirectory (named after the new
        classification), to allow for future use in training.

        Enter new classification for each image using numbers (e.g., 0/1/2) on keyboard, press enter to skip,
        press esc to stop reviewing the plate.

        :param threshold: Prediction threshold to identify inaccurate images
        :param save_dir: Directory to save re-classified images
        :return: List of indices of re-classified images
        """
        codes = {}
        for ascii_code, class_code in zip(range(48, 58), range(0, 10)):
            codes[ascii_code] = class_code
        skip_codes = {13: "enter"}
        stop_codes = {27: "esc"}
        codes.update(skip_codes)
        codes.update(stop_codes)

        inaccurate_images_indexes = self.get_inaccurate_images(threshold)
        changed_log = []
        for image_index in inaccurate_images_indexes:
            image, stamp = self.get_colony_image(image_index)
            i, j = image_index
            growth = self.growth_matrix[i][j]
            accuracy = self.accuracy_matrix[i][j]
            print()
            print(f"This image ({self.drug + str(self.concentration)} position {i} {j}) was labelled as {growth} "
                  f"with an accuracy of {accuracy * 100:.2f}")
            cv2.imshow(self.drug + str(self.concentration) + f" position {i} {j}", image)  # pylint: disable=no-member
            print("Press enter to continue, or enter new classification: ")
            while True:
                input_key = cv2.waitKey()  # pylint: disable=no-member
                if input_key not in codes:
                    print("Input not recognised, please try again..")
                    continue
                if input_key in stop_codes or input_key in skip_codes:
                    break
                try:
                    _ = self.get_key()[codes[input_key]]
                except IndexError:
                    print(f"Invalid input {codes[input_key]}: model key is {self.get_key()} [zero-indexed]")
                    continue
                break

            input_code = codes[input_key]

            if input_code in stop_codes.values():
                print(f"Stopping review for this plate: {self}.")
                break

            if input_code in skip_codes.values():
                print("Classification not changed.")
                continue

            if self.get_key()[input_code] == growth:
                print("Classification unchanged.")
                continue

            # reassign growth
            print(f"Reassigning image to {self.get_key()[input_code]}")
            self.growth_matrix[i][j] = self.get_key()[input_code]
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
                cv2.imwrite(save_path, image)  # pylint: disable=no-member
        return changed_log

    def convert_growth_codes(self, key: list[str]) -> list[list[str]]:
        """
        Convert growth codes to human-readable format using key
        E.g., [0, 1, 2] -> ["No growth", "Poor growth", "Good growth"]
        Sets self.growth_matrix

        :param key: List of growth codes (zero-indexed)
        :return: Growth matrix
        """
        self.growth_matrix = []
        for row in self.growth_code_matrix:
            _temp_row = []
            for code in row:
                _temp_row.append(key[code])
            self.growth_matrix.append(_temp_row)
        return self.growth_matrix

    def __repr__(self) -> str:
        return f"Plate of {self.drug} at {self.concentration}mg/L"

    def __lt__(self, other) -> bool:
        return self.concentration < other.concentration

    def __eq__(self, other) -> bool:
        return self.concentration == other.concentration and self.drug == other.drug

    def __gt__(self, other) -> bool:
        return self.concentration > other.concentration

    def __le__(self, other) -> bool:
        return self.concentration <= other.concentration

    def __ge__(self, other) -> bool:
        return self.concentration >= other.concentration

    def __ne__(self, other) -> bool:
        return self.concentration != other.concentration

    def __hash__(self) -> int:
        return hash((self.drug, self.concentration))


class PlateSet:
    def __init__(self, plates_list: list[Plate],
                 key: Optional[list[str]] = None) -> None:
        """
        Combines a list of Plate objects into a PlateSet to facilitate MIC calculation.
        Generally, plates would have a range of antimicrobial concentrations, including a control plate (concentration
        of 0.0mg/L).
        Plates must be annotated before initialisation (using Plate.annotate_images()) and have the same antimicrobial
        name and growth keys.

        :param plates_list: List of Plate objects
        """
        self.no_growth_key_items = None
        self.qc_matrix = None
        self.mic_matrix = None

        if key is not None:
            self.key = key
        else:
            _list_of_keys = []
            try:
                _list_of_keys = [i.get_key() for i in plates_list]
                self.key = _list_of_keys[0]
            except LookupError:
                warn("No key provided to PlateSet")
            if not all(i == _list_of_keys[0] for i in _list_of_keys):
                raise ValueError("Plates supplied to PlateSet have different growth keys")
            if not _list_of_keys:
                self.key = None

        drug_names = [i.drug for i in plates_list]
        if len(set(drug_names)) > 1:
            raise ValueError("Plates supplied to PlateSet have different antibiotic names")
        elif not len(set(drug_names)):
            raise ValueError("Plates supplied to PlateSet do not have antibiotic names")

        self.drug = plates_list[0].drug
        self.antibiotic_plates = [i for i in plates_list if i.concentration != 0.0]

        _temp_positive_control_plate = [i for i in plates_list if i.concentration == 0.0]
        if not _temp_positive_control_plate:
            warn(f"No control plate supplied to {self.drug} PlateSet")
        if len(_temp_positive_control_plate) > 1:
            warn(f"Multiple control plates supplied to {self.drug} PlateSet, control plates will be skipped.")
        else:
            [self.positive_control_plate] = _temp_positive_control_plate

        self.antibiotic_plates = sorted(self.antibiotic_plates)

        # check dimensions of plates' matrices
        if not self.valid_dimensions():
            raise ValueError("Plate matrices have different dimensions - unable to calculate MIC")

    def valid_dimensions(self) -> bool:
        """
        Check if all plates in PlateSet have the same x and y dimensions

        :return: True if all plates have the same dimensions, False otherwise
        """
        matrices_shapes = []
        for i in self.get_all_plates():
            try:
                matrices_shapes.append(i.matrix_dimensions(i.growth_code_matrix))
            except ValueError as e:
                raise ValueError(f"Plate {i} does not a matrix-shaped growth code matrix") from e

        return True if all(i == matrices_shapes[0] for i in matrices_shapes) else False

    def get_all_plates(self) -> list[Plate]:
        """
        Returns a sorted list of all plates in the PlateSet, including the control plate

        :return: List of Plate objects
        """
        return sorted(self.antibiotic_plates + [self.positive_control_plate])

    def convert_mic_matrix(self, mic_format: str = "string") -> np.array:
        """
        Converts format of MIC matrix

        :param mic_format: Format to convert to (only "string" is supported)
        :return: matrix (array) of MIC values
        """
        allowed_formats = ["string"]
        format_conversion = {"string": str}

        if mic_format not in allowed_formats:
            raise ValueError(f"MIC matrix formats must be one of: {allowed_formats}")
        output = self.mic_matrix.astype(format_conversion[mic_format])
        if mic_format == "string":
            max_mic_plate = max([i.concentration for i in self.antibiotic_plates])
            min_mic_plate = min([i.concentration for i in self.antibiotic_plates])
            for i, row in enumerate(output):
                for j, mic in enumerate(row):
                    if float(mic) > max_mic_plate:
                        output[i][j] = ">" + str(max_mic_plate)
                    elif float(mic) == min_mic_plate:
                        output[i][j] = "<" + str(min_mic_plate)
                    else:
                        output[i][j] = mic
        return output

    def calculate_mic(self, no_growth_key_items: tuple[int, ...]) -> np.array:
        """
        Calculate MIC matrix using image predictions.
        Sets self.mic_matrix

        :param no_growth_key_items: tuple of key items that should be classified as "no growth" for MIC purposes
        :return: MIC matrix
        """
        self.no_growth_key_items = no_growth_key_items
        self.antibiotic_plates = sorted(self.antibiotic_plates, reverse=True)
        max_concentration = max([i.concentration for i in self.antibiotic_plates]) * 2
        mic_matrix = np.array(self.antibiotic_plates[0].growth_code_matrix)
        mic_matrix = np.full(mic_matrix.shape, max([i.concentration for i in self.antibiotic_plates]) * 2)
        rows = range(mic_matrix.shape[0])
        cols = range(mic_matrix.shape[1])

        def get_first_negative_concentration(starting_concentration, i, j):
            if self.antibiotic_plates[0].growth_code_matrix[i][j] not in self.no_growth_key_items:
                return starting_concentration
            c = self.antibiotic_plates[0].concentration
            for plate in self.antibiotic_plates[1:]:
                if plate.growth_code_matrix[i][j] not in self.no_growth_key_items:
                    return c
                else:
                    c = plate.concentration
            return c

        for row in rows:
            for col in cols:
                mic_matrix[row][col] = get_first_negative_concentration(max_concentration, row, col)
        self.mic_matrix = mic_matrix
        return mic_matrix

    def generate_qc(self) -> np.array:
        """
        Generate QC matrix for PlateSet, as follows:

        "F" = FAIL - no growth in positive control plate, result should be disregarded
        "W" = WARNING - more than one change in concentration gradient. There should only be one change at the MIC
        breakpoint (where the images change from growth to no/poor growth). Depending on the application of the results,
        manual confirmation should be considered for warnings.
        "P" = PASS - no QC issues found

        :return: Matrix of QC values (strings)
        """
        if self.mic_matrix is None:
            raise ValueError(f"MIC matrix not found for {repr(self)} - please calculate MIC using calculate_mic()")
        qc_matrix = np.full(self.mic_matrix.shape, fill_value="", dtype=str)

        if self.positive_control_plate is None:
            warn(f"*Warning* - {repr(self)} does not contain a positive control plate.")
        else:
            for i, row in enumerate(self.positive_control_plate.growth_code_matrix):
                for j, item in enumerate(row):
                    if item in self.no_growth_key_items:
                        qc_matrix[i][j] = "F"
                    else:
                        qc_matrix[i][j] = "P"

        def simplify_codes(code):
            return 0 if code in self.no_growth_key_items else 1

        antibiotic_plates = sorted(self.antibiotic_plates, reverse=True)
        if len(antibiotic_plates) > 1:
            rows = range(qc_matrix.shape[0])
            cols = range(qc_matrix.shape[1])
            for i in rows:
                for j in cols:
                    if qc_matrix[i][j] == "F":
                        continue
                    previous_growth_code = simplify_codes(antibiotic_plates[0].growth_code_matrix[i][j])
                    flips = 0  # we only allow one "flip" from no growth -> growth
                    for k in antibiotic_plates[1:]:
                        next_growth_code = simplify_codes(k.growth_code_matrix[i][j])
                        if next_growth_code != previous_growth_code:
                            flips += 1
                        previous_growth_code = next_growth_code
                    if flips > 1:
                        qc_matrix[i][j] = "W"
        else:
            warn(f"*Warning* - {repr(self)} has insufficient plates for full QC")
        self.qc_matrix = qc_matrix
        return qc_matrix

    def review_poor_images(self, threshold: float = .9,
                           save_dir: Optional[str] = None) -> list[list[tuple[int, int]]]:
        """
        Review and re-classify images with prediction accuracy below threshold. Currently, supports up to 0--9 classes.
        If save_dir provided then colony images will also be saved to a subdirectory (named after the new
        classification), to allow for future use in training.

        Enter new classification for each image using 0/1/2 on keyboard, or press enter (or esc) to skip.

        :param threshold: Prediction threshold to identify inaccurate images
        :param save_dir: Directory to save re-classified images
        :return: List of indices of re-classified images
        """
        changed = [i.review_poor_images(threshold, save_dir) for i in self.get_all_plates()]
        n_changed = 0
        for i in changed:
            n_changed += len(i)
        print(f"{n_changed} images re-classified.")
        return changed

    def get_csv_data(self) -> list[dict]:
        """
        Get MIC and QC data in a format suitable for CSV export:
        List of dicts containing:
        - Antibiotic: Antibiotic name
        - Position: Position of the colony (e.g., A1, B2, etc.)
        - MIC: MIC value
        - QC: QC value (P, W, F)

        :return: List of dicts with MIC and QC data
        """
        if self.mic_matrix is None:
            raise ValueError("Please calculate MIC using PlateSet.calculate_mic() before exporting data")
        if self.qc_matrix is None:
            raise ValueError("Please generate QC using PlateSet.generate_qc() before exporting data")

        mic_matrix_str = self.convert_mic_matrix(mic_format="string")
        row_letters = ascii_uppercase[0:len(mic_matrix_str)]
        col_nums = [i + 1 for i in range(len(mic_matrix_str[0]))]
        output = []
        for i in range(len(row_letters)):
            for j in range(len(col_nums)):
                position = row_letters[i] + str(col_nums[j])
                mic = mic_matrix_str[i][j]
                qc = self.qc_matrix[i][j]
                output.append({'Antibiotic': self.drug, 'Position': position, 'MIC': mic, 'QC': qc})
        return output

    def __repr__(self) -> str:
        return (f"PlateSet of {self.drug} with {len(self.antibiotic_plates)} "
                f"concentrations: {[i.concentration for i in self.antibiotic_plates]}")


def plate_set_from_dir(path: Union[str, Path],
                       drug: str,
                       model: Model,
                       n_row: int = 8,
                       n_col: int = 12,
                       **kwargs) -> PlateSet:
    """
    Create a PlateSet from a directory of images. Images are annotated using the provided model.

    :param path: directory containing plate images (.jpg) with filenames indicating antibiotic concentration
    :param drug: name of drug
    :param model: model file to use for predictions
    :param n_row: number of rows in the plates
    :param n_col: number of columns in the plates
    :param kwargs: additional keyword arguments to pass to Plate constructor
    :return: PlateSet with MIC and QC values
    """
    image_paths = get_image_paths(path)
    plates = [Plate(drug,
                    concentration=get_concentration_from_path(i),
                    image=i,
                    model=model,
                    n_row=n_row,
                    n_col=n_col,
                    **kwargs) for i in image_paths]
    for i in plates:
        i.annotate_images()
    output = PlateSet(plates)
    return output
