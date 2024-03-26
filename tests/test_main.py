import subprocess
from tests.conftest import FIRST_LINE_MODEL_PATH, MIC_PLATES_PATH, DRUG_NAME, PROJECT_ROOT
import csv
import pytest
from os import path


@pytest.mark.assets_required
def test_main(tmp_path):
    d = tmp_path / "predictions"
    d.mkdir()

    # test that the main function runs without error
    result = subprocess.run(["python", path.join(PROJECT_ROOT, "src", "aigarmic", "main.py"),
                             "-m", FIRST_LINE_MODEL_PATH,
                             "-t", "binary",
                             "-o", d / "output.csv",
                             MIC_PLATES_PATH])
    assert result.returncode == 0
    assert (d / "output.csv").exists()

    with open(d / "output.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row)
            assert row['Antibiotic'] == DRUG_NAME
            assert "MIC" in row
            assert "Position" in row
            assert "QC" in row
            assert row["QC"] in ["P", "F", "W"]
