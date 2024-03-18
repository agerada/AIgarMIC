from src.aigarmic.process_plate_image import find_threshold_value, split_by_grid
from src.aigarmic.utils import get_image_paths
import cv2


test_images = get_image_paths("../images/ceftazidime/")


def test_find_threshold_value():
    for i in test_images:
        image = cv2.imread(i)
        blur = cv2.GaussianBlur(image, (25, 25), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        thresholds, n = find_threshold_value(gray)
        assert len(thresholds) == 96


def test_split_by_grid():
    for i in test_images:
        image = cv2.imread(i)
        split_by_grid(image, visualise_contours=False)

