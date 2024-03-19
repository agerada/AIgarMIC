import pytest

from src.aigarmic.plate import Plate, PlateSet, plate_set_from_dir
from src.aigarmic.utils import get_image_paths, get_conc_from_path
from os import path


IMAGES_PATH = "../images/"


@pytest.fixture
def plate_set_images():
    return get_image_paths(path.join(IMAGES_PATH, "ceftazidime"))


@pytest.fixture
def plates_list(plate_set_images):
    return [Plate("ceftazidime", get_conc_from_path(i), i, visualise_contours=False) for i in plate_set_images]


def test_plates(plates_list):
    assert [isinstance(i, Plate) for i in plates_list]

    sorted_plates = sorted(plates_list)
    assert sorted_plates[0].concentration == 0.0
    assert sorted_plates[-1].concentration == 64.0

