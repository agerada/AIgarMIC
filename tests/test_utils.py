from tests.conftest import MIC_PLATES_PATH, IMAGES_PATH
from src.aigarmic.utils import get_image_paths


def test_get_image_paths():
    for i in get_image_paths(MIC_PLATES_PATH):
        assert i.endswith(".jpg")

    print(get_image_paths(IMAGES_PATH))

    for paths in get_image_paths(IMAGES_PATH).values():
        for path in paths:
            assert path.endswith(".jpg")