import subprocess
from tests.conftest import FIRST_LINE_MODEL_PATH, MIC_PLATES_PATH, DRUG_NAME
import csv
import pytest


@pytest.mark.assets_required
def test_main(tmp_path):
    d = tmp_path / "predictions"
    d.mkdir()

    # test that the main function runs without error
    result = subprocess.run(["python", "../src/aigarmic/main.py", "-m", FIRST_LINE_MODEL_PATH,
                             "-t", "binary", "-o", d / "output.csv",
                             MIC_PLATES_PATH])
    assert result.returncode == 0
    assert (d / "output.csv").exists()

    with open(d / "output.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row['Antibiotic'] == DRUG_NAME
            assert "MIC" in row
            assert "Position" in row
            assert "QC" in row
            assert row["QC"] in ["P", "F", "W"]
