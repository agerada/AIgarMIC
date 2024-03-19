from tests.conftest import FIRST_LINE_MODEL_PATH


class TestBinaryModel:
    def test_predict(self, first_line_model, growth_image, no_growth_image):
        output_prediction = first_line_model.predict(growth_image)
        assert output_prediction['growth'] == "Growth"

        output_prediction = first_line_model.predict(no_growth_image)
        assert output_prediction['growth'] == "No growth"

    def test_load_model(self, first_line_model, growth_image):
        first_line_model.load_model(FIRST_LINE_MODEL_PATH)
        output_prediction = first_line_model.predict(growth_image)
        assert output_prediction['growth'] == "Growth"

    def test_get_key(self, first_line_model):
        assert first_line_model.get_key() == ["No growth", "Growth"]


class TestBinaryNestedModel:
    def test_predict(self, binary_nested_model,
                     growth_image,
                     no_growth_image,
                     poor_growth_image):
        output_prediction = binary_nested_model.predict(growth_image)
        assert output_prediction['growth'] == "Good growth"

        output_prediction = binary_nested_model.predict(no_growth_image)
        assert output_prediction['growth'] == "No growth"

        output_prediction = binary_nested_model.predict(poor_growth_image)
        assert output_prediction['growth'] == "Poor growth"
