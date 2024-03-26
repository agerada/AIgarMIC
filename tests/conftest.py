import os

import pytest
from aigarmic.model import BinaryModel, BinaryNestedModel
from aigarmic.plate import Plate
from aigarmic._img_utils import get_image_paths, get_concentration_from_path
from aigarmic.train import train_binary
from aigarmic._nn_design import model_design_spectrum_2024_binary_first_step
import cv2  # pylint: disable=import-error
from os import path

PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), os.pardir))
ASSETS_DIR = PROJECT_ROOT

FIRST_LINE_MODEL_PATH = path.join(ASSETS_DIR, "models", "spectrum_2024", "growth_no_growth")
SECOND_LINE_MODEL_PATH = path.join(ASSETS_DIR, "models", "spectrum_2024", "good_growth_poor_growth")
COLONY_IMAGE_PATH = path.join(ASSETS_DIR, "images", "single_colony.jpg")
NO_COLONY_IMAGE_PATH = path.join(ASSETS_DIR, "images", "single_no_growth.jpg")
POOR_GROWTH_IMAGE_PATH = path.join(ASSETS_DIR, "images", "single_poor_growth.jpg")
IMAGES_PATH = path.join(ASSETS_DIR, "images")
DRUG_NAME = "amikacin"
MIN_CONCENTRATION = 0.0
MAX_CONCENTRATION = 64.0
MIC_PLATES_PATH = path.join(IMAGES_PATH, "antimicrobials")
TARGET_MIC_CSV = path.join(MIC_PLATES_PATH, DRUG_NAME, "amikacin_target_spectrum_model.csv")
TRAIN_ANNOTATIONS_PATH = path.join(ASSETS_DIR, "images", "annotations", "train_binary")
TEST_ANNOTATIONS_PATH = path.join(ASSETS_DIR, "images", "annotations", "test_binary")
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 160


@pytest.fixture
def first_line_model_from_file():
    return BinaryModel(FIRST_LINE_MODEL_PATH,
                       trained_x=IMAGE_WIDTH, trained_y=IMAGE_HEIGHT,
                       threshold=0.5, key=["No growth", "Growth"])


@pytest.fixture
def second_line_model_from_file():
    return BinaryModel(SECOND_LINE_MODEL_PATH,
                       trained_x=IMAGE_WIDTH, trained_y=IMAGE_HEIGHT,
                       threshold=0.5, key=["Poor growth", "Good growth"])


@pytest.fixture
def binary_nested_model_from_file(first_line_model_from_file, second_line_model_from_file):
    return BinaryNestedModel(first_line_model_from_file, second_line_model_from_file,
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
    return get_image_paths(path.join(MIC_PLATES_PATH, DRUG_NAME))


@pytest.fixture
def plates_list(plates_images_paths):
    return [Plate(DRUG_NAME, get_concentration_from_path(i), i, visualise_contours=False) for i in plates_images_paths]


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


@pytest.fixture
def binary_model_trained():
    return train_binary(annotations_path=TRAIN_ANNOTATIONS_PATH,
                        model_design=model_design_spectrum_2024_binary_first_step(IMAGE_WIDTH, IMAGE_HEIGHT),
                        val_split=0.2,
                        image_width=IMAGE_WIDTH,
                        image_height=IMAGE_HEIGHT,
                        batch_size=2,
                        epochs=10)


