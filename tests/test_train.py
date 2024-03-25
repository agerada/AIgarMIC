from aigarmic.train import train_binary, train_softmax
from tests.conftest import TRAIN_ANNOTATIONS_PATH
from aigarmic.nn_design import model_design_spectrum_2024_binary_first_step
import pytest


@pytest.mark.assets_required
def test_train_binary(binary_model_trained):
    model, classes, history, results = binary_model_trained
    assert classes == ["0", "1"]
    assert "accuracy" in history.history
    assert "val_accuracy" in history.history
    assert "loss" in history.history
    assert "val_loss" in history.history


@pytest.mark.assets_required
def test_train_softmax():
    model, classes, history, results = train_softmax(annotations_path=TRAIN_ANNOTATIONS_PATH,
                 model_design=model_design_spectrum_2024_binary_first_step(160, 160))
    assert classes == ["0", "1"]
    assert "accuracy" in history.history
    assert "val_accuracy" in history.history
    assert "loss" in history.history
    assert "val_loss" in history.history

