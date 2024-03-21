from aigarmic.train import train_binary, train_softmax
from tests.conftest import TRAIN_ANNOTATIONS_PATH
from aigarmic.nn_design import model_design_spectrum_2024_binary_first_step


def test_train_binary():
    model, classes, history, results = train_binary(annotations_path=TRAIN_ANNOTATIONS_PATH,
                 model_design=model_design_spectrum_2024_binary_first_step(160, 160))
    assert classes == ["0", "1"]
    assert "accuracy" in history.history
    assert "val_accuracy" in history.history
    assert "loss" in history.history
    assert "val_loss" in history.history


def test_train_softmax():
    model, classes, history, results = train_softmax(annotations_path=TRAIN_ANNOTATIONS_PATH,
                 model_design=model_design_spectrum_2024_binary_first_step(160, 160))
    assert classes == ["0", "1"]
    assert "accuracy" in history.history
    assert "val_accuracy" in history.history
    assert "loss" in history.history
    assert "val_loss" in history.history
