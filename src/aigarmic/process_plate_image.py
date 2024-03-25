# Filename: 	process_plate_image.py
# Author: 	Alessandro Gerada
# Date: 	2023-01-27
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Pre-processing of agar plate images to extract individual colonies"""

import cv2
from imutils import contours
from typing import Union, Optional
from numpy import ndarray, empty


def find_threshold_value(image: ndarray, start: int = 20,
                         end: int = 100,
                         by: int = 1,
                         look_for: int = 96,
                         area_lower_bound: int = 1000) -> Optional[tuple[list, int]]:
    """
    Find threshold value that correctly splits an agar plate image into colony sub-images. Assumes that a black grid
    overlays the image.
    :param image: Image file loaded using cv2.imread
    :param start: starting threshold value
    :param end: ending threshold value
    :param by: threshold increment value
    :param look_for: target sub-images
    :param area_lower_bound: minimum area for a contour to be considered
    :return: tuple of contours and threshold value
    """
    for i in range(start, end, by):
        ret, thresh = cv2.threshold(image, i, 255, cv2.THRESH_BINARY_INV)

        # Find contours and filter using area
        _contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _contours = _contours[0] if len(_contours) == 2 else _contours[1]
        grid_contours = []
        for c in _contours:
            area = cv2.contourArea(c)
            if area > area_lower_bound:
                grid_contours.append(c)

        # sort contours and remove biggest (outer) grid square
        grid_contours = sorted(grid_contours, key=cv2.contourArea)
        grid_contours = grid_contours[:-1]

        # If we find the target boxes, return contours and threshold
        if len(grid_contours) == look_for:
            return grid_contours, i
    return None


def split_by_grid(image: ndarray, n_rows: int = 8,
                  visualise_contours: bool = False,
                  plate_name: Optional[str] = None) -> list[list[ndarray]]:
    """
    Split an agar plate image into individual colony sub-images using a grid overlay.

    :param image: image file loaded using cv2.imread
    :param n_rows: number of rows in the grid (columns will be inferred automatically)
    :param visualise_contours: if True, display the contours found (useful for validation)
    :param plate_name: name of plate to display in visualisation (useful for validation)
    :return: matrix of sub-images
    """
    if visualise_contours and not plate_name:
        raise ValueError("Pass plate name to split_by_grid if using visualise_contours")
    blur = cv2.GaussianBlur(image, (25, 25), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grid_contours, threshold_value = find_threshold_value(gray)

    if visualise_contours:
        _image = image
        cv2.drawContours(_image, grid_contours, -1, (0, 255, 0), 10)
        cv2.imshow(plate_name, _image)
        cv2.waitKey()

    if not grid_contours:
        raise ValueError("Unable to find contours threshold that returns correct number of colony images")
    else:
        print(f"Using {threshold_value} threshold value")

    # Sort contours, starting left to right
    (grid_contours, _) = contours.sort_contours(grid_contours, method="left-to-right")
    sorted_grid = []
    col = []  # temporary list to hold columns while sorting

    for (i, c) in enumerate(grid_contours, 1):
        col.append(c)
        if i % n_rows == 0:
            # found column - sort top to bottom and add to output
            (c_tmp, _) = contours.sort_contours(col, method="top-to-bottom")
            sorted_grid.append(c_tmp)
            col = []

    out_matrix = [[empty(shape=1) for _ in range(len(sorted_grid))] for _ in range(n_rows)]

    # Iterate through each box
    for j, col in enumerate(sorted_grid):
        for i, c in enumerate(col):
            x, y, w, h = cv2.boundingRect(c)
            cropped_image = image[y:y + h, x:x + w]
            out_matrix[i][j] = cropped_image

    return out_matrix
