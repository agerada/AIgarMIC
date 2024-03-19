from src.aigarmic.plate import Plate, PlateSet, plate_set_from_dir
from tests.conftest import DRUG_NAME, MIN_CONCENTRATION, MAX_CONCENTRATION, IMAGES_PATH, TARGET_MIC_CSV
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
    # MIC and QC values. Compare to results that were reported in Gerada et al 2024 Microbiology Spectrum paper.
    for target, prediction in zip(target_mic_values, plate_set.get_csv_data()):
        assert target == prediction
