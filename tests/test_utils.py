from tests.conftest import MIC_PLATES_PATH, IMAGES_PATH
from aigarmic.utils import get_image_paths, convertCV2toKeras


def test_get_image_paths():
    for i in get_image_paths(MIC_PLATES_PATH):
        assert i.endswith(".jpg")

    print(get_image_paths(IMAGES_PATH))

    for paths in get_image_paths(IMAGES_PATH).values():
        for path in paths:
            assert path.endswith(".jpg")


def test_convert_cv2to_keras(growth_image):
    image = convertCV2toKeras(growth_image)
    assert image.shape == (1, 160, 160, 3)
