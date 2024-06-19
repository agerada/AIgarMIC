import os
import subprocess
from tests.conftest import FIRST_LINE_MODEL_PATH, MIC_PLATES_PATH, DRUG_NAME, PROJECT_ROOT, TARGET_MIC_CSV
import csv
import pytest
from os import path


@pytest.mark.assets_required
def test_main(tmp_path):
    d = tmp_path / "predictions"
    d.mkdir()

    env = os.environ.copy()
    env["COVERAGE_PROCESS_START"] = ".coveragerc"

    # test that the main function runs without error
    # this just tests the running of the script, not model predictions, as that is tested
    # in test_plate.py
    result = subprocess.run(["coverage", "run", path.join(PROJECT_ROOT, "src", "aigarmic", "main.py"),
                             "-m", FIRST_LINE_MODEL_PATH,
                             "-t", "binary",
                             "-n", "0,1",
                             "-o", d / "output.csv",
                             MIC_PLATES_PATH],
                            env=env)
    assert result.returncode == 0
    assert (d / "output.csv").exists()

    target = {}
    with open(TARGET_MIC_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target[row["Position"]] = row["MIC"]

    with open(d / "output.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row)
            assert row['Antibiotic'] == DRUG_NAME
            assert "MIC" in row
            assert "Position" in row
            assert "QC" in row
            assert row["QC"] in ["P", "F", "W"]

            #assert row["MIC"] == target[row["Position"]]
