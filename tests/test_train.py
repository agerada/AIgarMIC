from aigarmic.train import train_binary, train_softmax
from tests.conftest import TRAIN_ANNOTATIONS_PATH
from aigarmic._nn_design import model_design_spectrum_2024_binary_first_step
import pytest


@pytest.mark.assets_required
def test_train_binary(binary_model_trained):
    # not checking performance here, just that a trained model is returned
    model, classes, history, results = binary_model_trained
    assert classes == ["0", "1"]
    assert "accuracy" in history.history
    assert "val_accuracy" in history.history
    assert "loss" in history.history
    assert "val_loss" in history.history


@pytest.mark.assets_required
def test_train_softmax():
    # not checking performance here, just that a trained model is returned
    model, classes, history, results = train_softmax(annotations_path=TRAIN_ANNOTATIONS_PATH,
                 model_design=model_design_spectrum_2024_binary_first_step(160, 160))
    assert classes == ["0", "1"]
    assert "accuracy" in history.history
    assert "val_accuracy" in history.history
    assert "loss" in history.history
    assert "val_loss" in history.history

