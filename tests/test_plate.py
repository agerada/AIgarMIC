from src.aigarmic.plate import Plate, PlateSet, plate_set_from_dir
from tests.conftest import DRUG_NAME, MIN_CONCENTRATION, MAX_CONCENTRATION, IMAGES_PATH, TARGET_MIC_CSV, basic_plates
from os import path
import numpy as np
import csv


def test_plates(plates_list):
    assert [isinstance(i, Plate) for i in plates_list]
    assert [i.drug == DRUG_NAME for i in plates_list]

    sorted_plates = sorted(plates_list)
    assert sorted_plates[0].concentration == MIN_CONCENTRATION
    assert sorted_plates[-1].concentration == MAX_CONCENTRATION

    temp_image, code = plates_list[0].get_colony_image()
    assert isinstance(temp_image, np.ndarray)
    assert isinstance(code, str)


def test_plate_set(binary_nested_model):
    plate_set = plate_set_from_dir(path=path.join(IMAGES_PATH, DRUG_NAME),
                                   drug=DRUG_NAME,
                                   model=binary_nested_model)
    assert isinstance(plate_set, PlateSet)

    with open(TARGET_MIC_CSV, "r", encoding='utf-8-sig') as f:
        target_mic_values = []
        reader = csv.DictReader(f)
        for line in reader:
            target_mic_values.append(line)

    # end-to-end validation, starting with images of multiple antibiotic concentrations, and ending with a
    # MIC and QC values. Compare to results that were reported in Gerada et al. 2024 Microbiology Spectrum paper.
    predicted_data = plate_set.get_csv_data()
    predicted_data = [{k: v for k, v in i.items() if k != "QC"} for i in predicted_data]
    for target, prediction in zip(target_mic_values, predicted_data):
        assert target == prediction


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
    basic_plate_set.calculate_mic()
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
