from tests.conftest import MIC_PLATES_PATH, IMAGES_PATH, DRUG_NAME
from aigarmic.img_utils import get_image_paths, convert_cv2_to_keras
import pytest
from os import path

deep_path = path.join(MIC_PLATES_PATH, DRUG_NAME)


@pytest.mark.assets_required
def test_get_image_paths():
    for i in get_image_paths(deep_path):
        assert i.endswith(".jpg")

    for paths in get_image_paths(IMAGES_PATH).values():
        for path in paths:
            assert path.endswith(".jpg")


@pytest.mark.assets_required
def test_convert_cv2to_keras(growth_image):
    image = convert_cv2_to_keras(growth_image)
    assert image.shape == (1, 160, 160, 3)
