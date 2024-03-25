from tests.conftest import FIRST_LINE_MODEL_PATH
import pytest


@pytest.mark.assets_required
class TestBinaryModel:
    def test_predict(self, first_line_model_from_file, growth_image, no_growth_image):
        output_prediction = first_line_model_from_file.predict(growth_image)
        assert output_prediction['growth'] == "Growth"

        output_prediction = first_line_model_from_file.predict(no_growth_image)
        assert output_prediction['growth'] == "No growth"

    def test_load_model(self, first_line_model_from_file, growth_image):
        first_line_model_from_file.load_model(FIRST_LINE_MODEL_PATH)
        output_prediction = first_line_model_from_file.predict(growth_image)
        assert output_prediction['growth'] == "Growth"

    def test_get_key(self, first_line_model_from_file):
        assert first_line_model_from_file.get_key() == ["No growth", "Growth"]


@pytest.mark.assets_required
class TestBinaryNestedModel:
    def test_predict(self, binary_nested_model_from_file,
                     growth_image,
                     no_growth_image,
                     poor_growth_image):
        output_prediction = binary_nested_model_from_file.predict(growth_image)
        assert output_prediction['growth'] == "Good growth"

        output_prediction = binary_nested_model_from_file.predict(no_growth_image)
        assert output_prediction['growth'] == "No growth"

        output_prediction = binary_nested_model_from_file.predict(poor_growth_image)
        assert output_prediction['growth'] == "Poor growth"
