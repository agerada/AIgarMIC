from aigarmic.process_plate_image import find_threshold_value, split_by_grid
from tests.conftest import plates_images_paths
import cv2  # pylint: disable=import-error
import pytest


@pytest.mark.assets_required
def test_find_threshold_value(plates_images_paths, growth_image):
    # successfully find a valid threshold that gives 96 small images
    for i in plates_images_paths:
        image = cv2.imread(i)
        blur = cv2.GaussianBlur(image, (25, 25), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        thresholds, n = find_threshold_value(gray)
        assert len(thresholds) == 96

    # fail, no grid on this image
    blur = cv2.GaussianBlur(growth_image, (25, 25), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    assert find_threshold_value(gray) is None


@pytest.mark.assets_required
def test_split_by_grid(plates_images_paths):
    for i in plates_images_paths:
        image = cv2.imread(i)
        split_images = split_by_grid(image, visualise_contours=False)
        assert len(split_images) == 8

        # check that each row has 12 columns (inferred from rows)
        assert len(split_images[0]) == 12
