from tests.conftest import TRAIN_ANNOTATIONS_PATH, TEST_ANNOTATIONS_PATH
from aigarmic.file_handlers import create_dataset_from_directory, predict_colony_images_from_directory, \
    save_training_log
from aigarmic._img_utils import get_image_paths
import tensorflow as tf
import pytest
import csv
from os import path


@pytest.mark.assets_required
def test_create_dataset_from_directory():
    no_growth_images = get_image_paths(path.join(TRAIN_ANNOTATIONS_PATH, "0"))
    growth_images = get_image_paths(path.join(TRAIN_ANNOTATIONS_PATH, "1"))

    train, val = create_dataset_from_directory(TRAIN_ANNOTATIONS_PATH,
                                               label_mode="binary",
                                               image_width=160,
                                               image_height=160,
                                               seed=12345,
                                               val_split=0.2,
                                               batch_size=32)
    assert isinstance(train, tf.data.Dataset)

    total_train = 0
    for i in train.unbatch():
        assert i[0].shape == (160, 160, 3)
        total_train += 1
    total_val = 0
    for i in val.unbatch():
        assert i[0].shape == (160, 160, 3)
        total_val += 1

    assert total_train + total_val == len(no_growth_images) + len(growth_images)
    assert total_val / (total_train + total_val) == pytest.approx(0.2, 0.05)


@pytest.mark.assets_required
def test_predict_colony_images_from_directory(binary_nested_model_from_file, tmp_path):
    d = tmp_path / "predictions"
    d.mkdir()
    predictions = predict_colony_images_from_directory(TEST_ANNOTATIONS_PATH,
                                                       model=binary_nested_model_from_file.first_line_model.keras_data,
                                                       class_names=["0", "1"],
                                                       image_width=160,
                                                       image_height=160,
                                                       save_path=d / "test_dataset_log.csv",
                                                       model_type="binary")

    for i in predictions:
        assert i['predicted_class'] == i['true_class']

    # check file save also correct
    with open(d / "test_dataset_log.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row['predicted_class'] == row['true_class']


@pytest.mark.assets_required
def test_save_training_log(binary_model_trained, tmp_path):
    model, classes, history, results = binary_model_trained
    d = tmp_path / "training_log"
    d.mkdir()
    save_training_log(history, d / "test_dataset_log.csv")

    with open(d / "test_dataset_log.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert float(row['loss']) >= 0
            assert float(row['accuracy']) >= 0
            assert float(row['val_loss']) >= 0
            assert float(row['val_accuracy']) >= 0
            assert float(row['accuracy']) <= 1
            assert float(row['val_accuracy']) <= 1
            assert int(row['epoch']) >= 0
