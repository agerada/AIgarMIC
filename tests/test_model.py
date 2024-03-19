import pytest

from src.aigarmic.model import BinaryModel, BinaryNestedModel
import cv2

FIRST_LINE_MODEL_PATH = "../models/spectrum_2024/growth_no_growth"
SECOND_LINE_MODEL_PATH = "../models/spectrum_2024/good_growth_poor_growth"
COLONY_IMAGE_PATH = "../images/single_colony.jpg"
NO_COLONY_IMAGE_PATH = "../images/single_no_growth.jpg"
POOR_GROWTH_IMAGE_PATH = "../images/single_poor_growth.jpg"


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


class TestBinaryModel:
    def test_predict(self, first_line_model, growth_image, no_growth_image):
        output_prediction = first_line_model.predict(growth_image)
        assert output_prediction['growth'] == "Growth"

        output_prediction = first_line_model.predict(no_growth_image)
        assert output_prediction['growth'] == "No growth"

    def test_load_model(self, first_line_model, growth_image):
        first_line_model.load_model(FIRST_LINE_MODEL_PATH)
        output_prediction = first_line_model.predict(growth_image)
        assert output_prediction['growth'] == "Growth"

    def test_get_key(self, first_line_model):
        assert first_line_model.get_key() == ["No growth", "Growth"]


class TestBinaryNestedModel:
    def test_predict(self, binary_nested_model,
                     growth_image,
                     no_growth_image,
                     poor_growth_image):
        output_prediction = binary_nested_model.predict(growth_image)
        assert output_prediction['growth'] == "Good growth"

        output_prediction = binary_nested_model.predict(no_growth_image)
        assert output_prediction['growth'] == "No growth"

        output_prediction = binary_nested_model.predict(poor_growth_image)
        assert output_prediction['growth'] == "Poor growth"
