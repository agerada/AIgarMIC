import pytest
from src.aigarmic.model import BinaryModel, BinaryNestedModel
from src.aigarmic.plate import Plate
from src.aigarmic.utils import get_image_paths, get_conc_from_path
import cv2
from os import path

FIRST_LINE_MODEL_PATH = "../models/spectrum_2024/growth_no_growth"
SECOND_LINE_MODEL_PATH = "../models/spectrum_2024/good_growth_poor_growth"
COLONY_IMAGE_PATH = "../images/single_colony.jpg"
NO_COLONY_IMAGE_PATH = "../images/single_no_growth.jpg"
POOR_GROWTH_IMAGE_PATH = "../images/single_poor_growth.jpg"
IMAGES_PATH = "../images/"
DRUG_NAME = "amikacin"
MIN_CONCENTRATION = 0.0
MAX_CONCENTRATION = 64.0
TARGET_MIC_CSV = "../images/amikacin/amikacin_target_spectrum_model.csv"
MIC_PLATES_PATH = path.join(IMAGES_PATH, DRUG_NAME)


@pytest.fixture
def first_line_model():
    return BinaryModel(FIRST_LINE_MODEL_PATH,
                       trained_x=160, trained_y=160,
                       threshold=0.5, key=["No growth", "Growth"])


@pytest.fixture
def second_line_model():
    return BinaryModel(SECOND_LINE_MODEL_PATH,
                       trained_x=160, trained_y=160,
                       threshold=0.5, key=["Poor growth", "Good growth"])


@pytest.fixture
def binary_nested_model(first_line_model, second_line_model):
    return BinaryNestedModel(first_line_model, second_line_model,
                             first_model_accuracy_acceptance=0.9,
                             suppress_first_model_accuracy_check=True)


@pytest.fixture
def growth_image():
    return cv2.imread(COLONY_IMAGE_PATH)


@pytest.fixture
def no_growth_image():
    return cv2.imread(NO_COLONY_IMAGE_PATH)


@pytest.fixture
def poor_growth_image():
    return cv2.imread(POOR_GROWTH_IMAGE_PATH)


@pytest.fixture
def plates_images_paths():
    return get_image_paths(MIC_PLATES_PATH)


@pytest.fixture
def plates_list(plates_images_paths):
    return [Plate(DRUG_NAME, get_conc_from_path(i), i, visualise_contours=False) for i in plates_images_paths]


@pytest.fixture
def basic_plates():
    """
    Creates a basic set of plates without using images, to support testing of PlateSet logic.
    """
    output = [
        Plate('genta', 128.),
        Plate('genta', 64.),
        Plate('genta', 32.),
        Plate('genta', 16.),
        Plate('genta', 0.),
    ]

    output[0].growth_code_matrix = [
        [0, 2],
        [0, 0]]
    output[1].growth_code_matrix = [
        [1, 2],
        [2, 0]]
    output[2].growth_code_matrix = [
        [2, 2],
        [1, 0]]
    output[3].growth_code_matrix = [
        [2, 2],
        [2, 0]]

    output[4].growth_code_matrix = [
        [2, 2],
        [2, 0]]

    return output
