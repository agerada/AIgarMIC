from aigarmic.plate import Plate, PlateSet, plate_set_from_dir
from tests.conftest import (DRUG_NAME,
                            MIN_CONCENTRATION,
                            MAX_CONCENTRATION,
                            TARGET_MIC_CSV,
                            MIC_PLATES_PATH,
                            basic_plates,
                            target_mic_spectrum)
from os import path
import numpy as np
import csv
import pytest


@pytest.mark.assets_required
def test_plates(plates_list):
    # check plates correctly instantialised from images
    assert [isinstance(i, Plate) for i in plates_list]
    assert [i.drug == DRUG_NAME for i in plates_list]

    # check that plates sorted correctly (by concentration)
    # indirectly checks implementation of dunder methods such as __lt__
    sorted_plates = sorted(plates_list)
    assert sorted_plates[0].concentration == MIN_CONCENTRATION
    assert sorted_plates[-1].concentration == MAX_CONCENTRATION

    # can fetch a random colony image
    temp_image, code = plates_list[0].get_colony_image()
    assert isinstance(temp_image, np.ndarray)
    assert isinstance(code, str)


@pytest.mark.assets_required
def test_annotate_images(plates_list, binary_nested_model_from_file):
    # checks that an agar plate image is correctly annotated
    [single_plate] = [i for i in plates_list if i.concentration == MIN_CONCENTRATION]
    single_plate.annotate_images(model=binary_nested_model_from_file)
    assert single_plate.growth_code_matrix[1][0] == 2
    assert single_plate.growth_code_matrix[0][1] == 2


@pytest.mark.assets_required
def test_implicit_annotate_images(plates_images_paths, binary_nested_model_from_file):
    # providing an image path and model should automatically annotate the image
    plate = Plate(drug=DRUG_NAME,
                  concentration=MIN_CONCENTRATION,
                  image=plates_images_paths[0],
                  n_row=8,
                  n_col=12,
                  model=binary_nested_model_from_file)
    assert plate.image_matrix is not None
    for row in plate.image_matrix:
        for item in row:
            assert isinstance(item, np.ndarray)

    assert plate.growth_code_matrix is not None
    for row in plate.growth_code_matrix:
        for item in row:
            assert isinstance(item, int)

    # omitting the model should not annotate the image
    plate = Plate(drug=DRUG_NAME,
                  concentration=MIN_CONCENTRATION,
                  n_row=8,
                  n_col=12,
                  image=plates_images_paths[0])
    assert plate.growth_code_matrix is None


@pytest.mark.assets_required
def test_plate_set(binary_nested_model_from_file, target_mic_spectrum):
    plate_set = plate_set_from_dir(path=path.join(MIC_PLATES_PATH, DRUG_NAME),
                                   drug=DRUG_NAME,
                                   model=binary_nested_model_from_file)
    assert isinstance(plate_set, PlateSet)

    plate_set.calculate_mic(no_growth_key_items=tuple([0, 1]))
    plate_set.generate_qc()

    # end-to-end validation, starting with images of multiple antibiotic concentrations, and ending with a
    # MIC and QC values. Compare to results that were reported in Gerada et al. 2024 Microbiology Spectrum paper.
    predicted_data = plate_set.get_csv_data()
    predicted_data = [{k: v for k, v in i.items() if k != "QC"} for i in predicted_data]
    for prediction in predicted_data:
        assert prediction["MIC"] == target_mic_spectrum[prediction["Position"]]


def test_convert_growth_codes(basic_plates):
    for i in basic_plates:
        i.convert_growth_codes(key=["No growth", "Poor Growth", "Growth"])

    target_growths = [[]] * len(basic_plates)

    target_growths[0] = [
        ["No growth", "Growth"],
        ["No growth", "No growth"]
    ]
    target_growths[1] = [
        ["Poor Growth", "Growth"],
        ["Growth", "No Growth"]
    ]
    target_growths[2] = [
        ["Growth", "Growth"],
        ["No Growth", "No Growth"]
    ]
    target_growths[3] = [
        ["Growth", "Growth"],
        ["Growth", "No Growth"]
    ]
    target_growths[4] = [
        ["Growth", "Growth"],
        ["Growth", "Growth"]
    ]
    assert [i.growth_matrix == j for i, j in zip(basic_plates, target_growths)]


def test_calculate_mic(basic_plates):
    _key = ["No growth", "Poor Growth", "Growth"]
    for i in basic_plates:
        i.set_key(_key)
        i.convert_growth_codes(key=_key)
    basic_plate_set = PlateSet(basic_plates)
    basic_plate_set.calculate_mic(no_growth_key_items=tuple([0, 1]))
    basic_plate_set.generate_qc()

    target_mic = [["64.0", ">128.0"],
                  ["128.0", "<16.0"]]
    target_qc = [["P", "P"],
                 ["W", "F"]]
    target_output = [
        {'Antibiotic': 'genta', 'Position': 'A1', 'MIC': target_mic[0][0], 'QC': target_qc[0][0]},
        {'Antibiotic': 'genta', 'Position': 'A2', 'MIC': target_mic[0][1], 'QC': target_qc[0][1]},
        {'Antibiotic': 'genta', 'Position': 'B1', 'MIC': target_mic[1][0], 'QC': target_qc[1][0]},
        {'Antibiotic': 'genta', 'Position': 'B2', 'MIC': target_mic[1][1], 'QC': target_qc[1][1]}
    ]
    assert basic_plate_set.get_csv_data() == target_output


def test_generate_qc():
    test_qc_plates = [
        Plate('genta', 64.),
        Plate('genta', 32.),
        Plate('genta', 16.),
        Plate('genta', 0.),
    ]

    test_qc_plates[0].add_growth_code_matrix([
        [3, 2],
        [0, 1]])
    test_qc_plates[1].add_growth_code_matrix([
        [0, 2],
        [3, 2]])
    test_qc_plates[2].add_growth_code_matrix([
        [3, 2],
        [2, 3]])
    test_qc_plates[3].add_growth_code_matrix([
        [3, 3],
        [3, 2]])

    test_qc_plate_set = PlateSet(test_qc_plates)
    test_qc_plate_set.calculate_mic(no_growth_key_items=(0, 1))
    qc_test = test_qc_plate_set.generate_qc()
    qc_target = [['W', 'P'],
                 ['P', 'P']]
    for row_test, row_target in zip(qc_test, qc_target):
        for col_test, col_target in zip(row_test, row_target):
            assert col_test == col_target

    test_qc_plate_set.calculate_mic(no_growth_key_items=(0, 1, 2))
    qc_test = test_qc_plate_set.generate_qc()
    qc_target = [['W', 'P'],
                 ['W', 'F']]
    for row_test, row_target in zip(qc_test, qc_target):
        for col_test, col_target in zip(row_test, row_target):
            assert col_test == col_target


def test_valid_dimensions():
    test_dimension_plates = [
        Plate('genta', 64.),
        Plate('genta', 0.),
    ]

    #  fails because the growth_code_matrix is not of the same dimensions:
    test_dimension_plates[0].growth_code_matrix = [
        [0, 2],
        [0, 0]]
    test_dimension_plates[1].growth_code_matrix = [
        [1, 2, 2],
        [2, 2, 2]]

    with pytest.raises(ValueError):
        PlateSet(test_dimension_plates)

    #  fails because the growth_code_matrix are not valid matrix dimensions,
    #  despite being the same.
    test_dimension_plates[0].growth_code_matrix = [
        [0, 2],
        [0]]
    test_dimension_plates[1].growth_code_matrix = [
        [1, 2, 2],
        [2]]

    with pytest.raises(ValueError):
        PlateSet(test_dimension_plates)


@pytest.mark.assets_required
def test_get_colony_image(plates_list):
    indices = [(5, 5), (7, 10), (1, 1), (2, 6), (3, 1)]
    [single_plate] = [i for i in plates_list if i.concentration == MAX_CONCENTRATION]
    for i in indices:
        temp_image, code = single_plate.get_colony_image(i)
        assert isinstance(temp_image, np.ndarray)
        assert isinstance(code, str)
        assert code == "_".join([DRUG_NAME,
                                 str(single_plate.concentration),
                                 "i", str(i[0]),
                                 "j", str(i[1])])

    # out of bounds check
    with pytest.raises(IndexError):
        single_plate.get_colony_image((12, 14))


def test_plate_with_growth_code():
    test_plate = Plate('genta', 64.,
                       growth_code_matrix=[[0, 2],
                                           [1, 0]],
                       n_col=2, n_row=2)
    assert test_plate.growth_code_matrix[0][0] == 0
    assert test_plate.growth_code_matrix[1][1] == 0
    assert test_plate.growth_code_matrix[0][1] == 2
    assert test_plate.growth_code_matrix[1][0] == 1

    # invalid matrix dimensions raises error
    with pytest.raises(ValueError):
        test_plate.add_growth_code_matrix([[0, 2],
                                           [1, 0],
                                           [1, 0, 2]])

    with pytest.raises(ValueError):
        # more than two-dimensional matrix is not allowed
        test_plate.add_growth_code_matrix([
            [[0, 2],
             [0, 2],
             [0, 2],
             [1, 0]
             ]
        ])

    with pytest.raises(ValueError):
        # negative growth codes are not allowed
        test_plate.add_growth_code_matrix([[-1, 2],
                                           [1, 0]])

    with pytest.raises(ValueError):
        # growth codes must be integers
        test_plate.add_growth_code_matrix([[0., 2.],
                                           [1., 3.]])

    # test provision of appropriate key does not raise error
    _ = Plate('genta', 64.,
              growth_code_matrix=[[0, 1],
                                  [1, 0]],
              key=["No growth", "Growth"],)

    with pytest.raises(ValueError):
        # growth codes cannot exceed the length of the key
        _ = Plate('genta', 64.,
                  growth_code_matrix=[[0, 2],
                                      [1, 0]],
                  key=["No growth", "Growth"],)
