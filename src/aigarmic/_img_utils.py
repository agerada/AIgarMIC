# Filename:      _img_utils.py
# Author:        Alessandro Gerada
# Date:          01/10/2022
# Copyright:     Alessandro Gerada 2022
# Email:         alessandro.gerada@liverpool.ac.uk

"""
Image utility functions
"""
from os import remove
from typing import Union
import cv2  # pylint: disable=import-error
import numpy as np
import os
import tensorflow as tf
from pathlib import Path


def convert_cv2_to_keras(image, size_x=160, size_y=160) -> np.ndarray:
    """
    Convert a cv2 image to a keras image

    :param image: Image loaded using cv2.imread
    :param size_x: Width to resize image to (pixels)
    :param size_y: Height to resize image to (pixels)
    :return: Image as a numpy array
    """
    # resize
    image = cv2.resize(image, (size_x, size_y))
    # convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape(1, size_x, size_y, 3)
    image = image.astype(np.float32)
    return image


def get_concentration_from_path(path: Union[str, Path]) -> float:
    """
    get concentration from plate image path, e.g.
    antibiotic1/0.125.jpg -> 0.125

    :param path: Path to plate image
    :return: Concentration
    """
    split_text = os.path.split(path)
    split_text = split_text[-1]
    concentration_str = os.path.splitext(split_text)[0]
    return float(concentration_str)


def get_paths_from_directory(path: Union[str, Path]) -> dict[str, list[str]]:
    """
    Returns a dict of abx_names: [image1_path, image2_path, etc.]
    If there are no antibiotic subdirectories, "unnamed" is used 
    for abx_names (length = 1)

    :param path: Path to directory containing antibiotic subdirectories
    :return: dict of abx_names: [image1_path, image2_path, etc.]
    """
    abx_names = [i for i in os.listdir(path) if not i.startswith('.') and os.path.isdir(os.path.join(path, i))]
    
    if not abx_names: 
        abx_names = [""]
    
    plate_images_paths = {}
    for abx in abx_names: 
        _path = os.path.join(path, abx)
        _temp_plate_images_paths = os.listdir(_path)
        _temp_plate_images_paths = [i for i in _temp_plate_images_paths if i.count('.jpg') > 0 or i.count('.JPG') > 0]
        _temp_plate_images_paths = [os.path.join(path, abx, i) for i in _temp_plate_images_paths]
        plate_images_paths[abx] = _temp_plate_images_paths

    return plate_images_paths


def keras_image_to_cv2(image: tf.Tensor) -> np.ndarray:
    """
    Convert a keras image to a cv2 image

    :param image: Image as a tensor
    :return: Image as a numpy array
    """
    img = image.numpy().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_image_paths(dir_path,
                    extensions: tuple[str, ...] = ('.jpg', '.JPG')) -> Union[list[str], dict[str, list[str]]]:
    """
    If there are no subdirectories in dir, returns a list of image paths
    If there are subdirectories, returns a dict of 'subdir_name': 'path'

    :param dir_path: Path to directory containing images
    :param extensions: Tuple of image extensions to search for
    :return: List of image paths or dict of 'subdir_name': 'path'
    """
    sub_dirs = [i for i in os.listdir(dir_path) if not i.startswith('.') and os.path.isdir(os.path.join(dir_path, i))]
    
    if not sub_dirs:
        return [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith(tuple(extensions))]
    else: 
        output = {}
        for i in sub_dirs:
            _parent_path = os.path.join(dir_path, i)
            _temp_image_paths = os.listdir(_parent_path)
            _temp_image_paths = [i for i in _temp_image_paths if i.endswith(tuple(extensions))]
            _temp_image_paths = [os.path.join(_parent_path, i) for i in _temp_image_paths]
            output[i] = _temp_image_paths
        return output


class Deleter:
    def __init__(self, confirm=True):
        self.confirm = confirm
        self.converter = {
            "y": "yes",
            "ye": "yes",
            "yes": "yes",
            "n": "no",
            "no": "no",
            "a": "all",
            "al": "all",
            "all": "all"
        }

    def delete_file(self, file):
        if not self.confirm:
            print(f"Deleting duplicate file: {file}")
            remove(file)
        else:
            while True:
                user_in = input(f"[ Yes / No / All ] Delete duplicate file: {file}").lower()
                if user_in not in self.converter:
                    print("Input not recognised, please repeat")
                    continue
                if self.converter[user_in] == "yes":
                    print(f"Deleting file: {file}")
                    remove(file)
                    break
                if self.converter[user_in] == "no":
                    print(f"Skipping file: {file}")
                    break
                if self.converter[user_in] == "all":
                    print("Warnings suppressed")
                    print(f"Deleting file: {file}")
                    remove(file)
                    self.confirm = False
                    break
                else:
                    raise LookupError("Unable to interpret user loop")


def is_similar(image1, image2):
    """
    From: https://stackoverflow.com/a/23199159
    """
    return image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any())


def in_list(image, target_list):
    if not target_list:
        return False
    else:
        for i in target_list:
            if is_similar(i, image):
                return True
        return False


def rename_jpg(file_in, file_out):
    if not file_in or not file_out:
        return
    print(f"Renaming {file_in} to {file_out}")
    os.rename(file_in, file_out)
