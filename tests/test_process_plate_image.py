from src.aigarmic.process_plate_image import find_threshold_value, split_by_grid
from src.aigarmic.utils import get_image_paths
import cv2
from os import path

IMAGES_PATH = "../images/"

test_images = get_image_paths(path.join(IMAGES_PATH, "ceftazidime"))
single_colony_image = cv2.imread(path.join(IMAGES_PATH, "single_colony.jpg"))


def test_find_threshold_value():
    for i in test_images:
        image = cv2.imread(i)
        blur = cv2.GaussianBlur(image, (25, 25), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        thresholds, n = find_threshold_value(gray)
        assert len(thresholds) == 96

    # fail, no grid on this image
    blur = cv2.GaussianBlur(single_colony_image, (25, 25), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    assert find_threshold_value(gray) is None


def test_split_by_grid():
    for i in test_images:
        image = cv2.imread(i)
        split_images = split_by_grid(image, visualise_contours=False)
        assert len(split_images) == 8
        assert len(split_images[0]) == 12

