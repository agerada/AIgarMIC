#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: 	model.py
# Author: 	Alessandro Gerada
# Date: 	2023-03-15
# Copyright: 	Alessandro Gerada 2023
# Email: 	alessandro.gerada@liverpool.ac.uk

"""Implementation of Model classes"""

import tensorflow as tf
import numpy as np
from src.aigarmic.utils import convertCV2toKeras
from typing import Optional


class Model:
    def __init__(self, path: str,
                 trained_x: int,
                 trained_y: int,
                 key: Optional[list[str]]) -> None:
        """
        Base class for keras model to interpret colony image growth. Optionally provide a key to convert model output
        to a growth label, e.g.,
        ['No growth', 'Growth'] -> predictions of 0 will be interpreted as 'No growth', 1 as 'Growth'.
        ['No growth', 'Poor growth', 'Good growth'] -> predictions of 0 will be interpreted as 'No growth',
        1 as 'Poor growth', 2 as 'Good growth'.

        :param path: path to saved model
        :param trained_x: image width used to train model
        :param trained_y: image height used to train model
        :param key: key to interpret model output
        """
        self.path = path
        self.keras_data = None
        self.load_model(self.path)
        if key:
            self.key = key
        else:
            # infer key from final output layer of loaded model
            final_layer = self.keras_data.layers[-1]
            inferred_key = [str(i) for i in range(final_layer.output.shape[1])]
            self.key = inferred_key
        self.trained_x = trained_x
        self.trained_y = trained_y

    def load_model(self, path: str) -> None:
        """
        Load a keras model from file

        :param path: path to saved model
        """
        self.keras_data = tf.keras.models.load_model(path)

    def get_key(self) -> list[str]:
        """
        Return key to convert model output to human-readable label
        :return:
        """
        if not self.key:
            raise LookupError(
                """
                Unable to find an interpretation key to convert scores to label. Please provide one
                on Model class construction
                """
            )
        else:
            return self.key

    def predict(self, image):
        raise NotImplementedError


class SoftmaxModel(Model):
    """
    SoftmaxModel is a one-stop model when more than one growth category is present, e.g.:
    ['No growth', 'Poor growth', 'Good growth']
    """

    def predict(self, image: np.ndarray) -> dict:
        """
        Predict growth category from image
        :param image: image loaded using cv2.imread
        :return: dictionary with keys 'prediction', 'score', 'growth_code', 'growth', 'accuracy'
        """
        key = self.get_key()
        image = convertCV2toKeras(image, size_x=self.trained_x, size_y=self.trained_y)
        output = {'prediction': self.keras_data.predict(image)}
        output['score'] = tf.nn.softmax(output['prediction'])
        output['growth_code'] = np.argmax(output['score'])
        output['growth'] = key[output['growth_code']]
        output['accuracy'] = np.max(output['score'])
        return output


class BinaryModel(Model):
    def __init__(self, path: str,
                 trained_x: int,
                 trained_y: int,
                 key: Optional[list[str]],
                 threshold: float = 0.5):
        """
        BinaryModel is a one-stop model when only two growth categories are present, e.g.:
        ['No growth', 'Growth'], or can be used in a two-step model:
        ['No growth', 'Growth'] -> ['Poor growth', 'Good growth']
        :param path: path to saved model
        :param trained_x: width of image used to train model
        :param trained_y: height of image used to train model
        :param key: key to interpret model output
        :param threshold: binary threshold to convert model output to growth code
        """
        if len(key) != 2:
            raise ValueError("Key of object of class BinaryModel must have length 2")
        if threshold < 0 or threshold > 1:
            raise ValueError("Binary threshold must be between 0 and 1")
        self.threshold = threshold
        super().__init__(path=path, key=key, trained_x=trained_x, trained_y=trained_y)

    def predict(self, image: np.ndarray) -> dict:
        """
        Predict growth category from image

        :param image: image loaded using cv2.imread
        :return: dictionary with keys 'prediction', 'score', 'growth_code', 'growth', 'accuracy'
        """
        key = self.get_key()
        image = convertCV2toKeras(image, size_x=self.trained_x, size_y=self.trained_y)
        prediction = self.keras_data.predict(image)
        [prediction] = prediction.reshape(-1)
        output = {'prediction': prediction}
        output['score'] = 0 if output['prediction'] <= self.threshold else 1
        output['growth_code'] = output['score']
        output['growth'] = key[output['growth_code']]
        output['accuracy'] = prediction if output['score'] == 1 else 1 - prediction
        return output


class BinaryNestedModel:
    def __init__(self, first_line_model: BinaryModel,
                 second_line_model: BinaryModel,
                 first_model_accuracy_acceptance: float = 0.9,
                 suppress_first_model_accuracy_check: bool = False) -> None:
        """
        Converts two BinaryModels into a two-step model. Generally this is used to support a more complex model for the
        second step of growth classification, which is generally more technically challenging. This approach also allows
        for computational efficiency by not running the second model in some situations (e.g., poor performance on the
        first model). Note that the key is inherited from the base models.

        :param first_line_model: BinaryModel object (generally differentiates growth from no growth)
        :param second_line_model: BinaryModel object (generally differentiates poor growth from good growth)
        :param first_model_accuracy_acceptance: minimum accuracy of first model to proceed to second model
        :param suppress_first_model_accuracy_check: if True, always proceed to second model
        """
        self.first_line_model = first_line_model
        self.second_line_model = second_line_model
        self.first_model_accuracy_acceptance = first_model_accuracy_acceptance
        self.suppress_first_model_accuracy_check = suppress_first_model_accuracy_check

        _key = self.second_line_model.get_key()
        _key.insert(0, self.first_line_model.get_key()[0])
        self.key = _key

    def predict(self, image):
        first_line_classification = self.first_line_model.predict(image)
        if not self.suppress_first_model_accuracy_check and \
                first_line_classification['accuracy'] < self.first_model_accuracy_acceptance:
            # Do not try to make any additional predictions if first model poor
            return first_line_classification
        elif first_line_classification['growth_code'] == 0:
            return first_line_classification
        else:
            second_line_classification = self.second_line_model.predict(image)
            # Binary model assigns growth code 0 or 1, therefore need to 
            # correct second_line_classification['growth_code']
            second_line_classification['growth_code'] += 1
            second_line_classification['growth'] = self.second_line_model.get_key()[
                second_line_classification['growth_code']]
            return second_line_classification

    def get_key(self):
        return self.key
