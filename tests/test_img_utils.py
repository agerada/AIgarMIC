import shutil
from tests.conftest import COLONY_IMAGE_PATH, NO_COLONY_IMAGE_PATH
from aigarmic._img_utils import Deleter, is_similar, in_list
from os import path
import cv2  # pylint: disable=import-error


def test_delete_file(tmp_path):
    temp_file = tmp_path / "test_image.jpg"
    assert not path.exists(temp_file)
    shutil.copy(COLONY_IMAGE_PATH, temp_file)
    assert path.exists(temp_file)
    d = Deleter(confirm=False)
    d.delete_file(temp_file)
    assert not path.exists(temp_file)


def test_is_similar(tmp_path):
    image_1_copy_1 = tmp_path / "copy1.jpg"
    shutil.copy(COLONY_IMAGE_PATH, image_1_copy_1)

    image_1_copy_2 = tmp_path / "copy2.jpg"
    shutil.copy(COLONY_IMAGE_PATH, image_1_copy_2)

    assert is_similar(cv2.imread(str(image_1_copy_1)), cv2.imread(str(image_1_copy_2)))

    image_2 = tmp_path / "copy3.jog"
    shutil.copy(NO_COLONY_IMAGE_PATH, image_2)

    assert not is_similar(cv2.imread(str(image_1_copy_2)), cv2.imread(str(image_2)))


def test_in_list():
    img_list = list()
    img_list.append(cv2.imread(COLONY_IMAGE_PATH))

    assert in_list(cv2.imread(COLONY_IMAGE_PATH), img_list)

    assert not in_list(cv2.imread(NO_COLONY_IMAGE_PATH), img_list)
