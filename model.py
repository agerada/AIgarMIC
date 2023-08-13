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
from utils import convertCV2toKeras

class Model(): 
    def __init__(self, path, trained_x, trained_y, key=None): 
        """
        trained_x and trained_y are the image width and height on which the neural network
        was trained
        """
        self.path = path
        self.load_model(self.path)
        if key:
            self.key = key
        else:
            # infer key from final output layer of loaded model
            final_layer = self.keras_data.layers[-1]
            inferred_key = [str(i) for i in range(final_layer.output.shape[1])]
            self.key = tuple(inferred_key)
        self.trained_x = trained_x
        self.trained_y = trained_y
    
    def load_model(self, path): 
        self.keras_data = tf.keras.models.load_model(path)

    def get_key(self): 
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
    def predict(self, image): 
        key = self.get_key()
        image = convertCV2toKeras(image, size_x=self.trained_x, size_y=self.trained_y)
        output = {}
        output['prediction'] = self.keras_data.predict(image)
        output['score'] = tf.nn.softmax(output['prediction'])
        output['growth_code'] = np.argmax(output['score'])
        output['growth'] = key[output['growth_code']]
        output['accuracy'] = np.max(output['score'])
        return output

class BinaryModel(Model): 
    def __init__(self, path, key, trained_x, trained_y, threshold = 0.5): 
        if len(key) != 2: 
            raise ValueError("Key of object of class BinaryModel must have length 2")
        self.threshold = threshold
        super().__init__(path=path, key=key, trained_x=trained_x, trained_y=trained_y)

    def predict(self, image): 
        key = self.get_key()
        image = convertCV2toKeras(image, size_x=self.trained_x, size_y=self.trained_y)
        output = {}
        prediction = self.keras_data.predict(image)
        [prediction] = prediction.reshape(-1)
        output['prediction'] = prediction
        output['score'] = 0 if output['prediction'] <= self.threshold else 1
        output['growth_code'] = output['score']
        output['growth'] = key[output['growth_code']]
        output['accuracy'] = prediction if output['score'] == 1 else 1 - prediction
        return output
    
class BinaryNestedModel: 
    """
    Pass two instances of class Model as input. There is no need to run Model.predict()
    prior to use in this class.
    """
    def __init__(self, first_line_model: Model, second_line_model: Model, 
                 first_model_accuracy_acceptance=0.9,
                 suppress_first_model_accuracy_check=False): 
        self.first_line_model = first_line_model
        self.second_line_model = second_line_model
        self.first_model_accuracy_acceptance = first_model_accuracy_acceptance
        self.suppress_first_model_accuracy_check = suppress_first_model_accuracy_check

        _key = self.second_line_model.get_key()
        _key.insert(0, self.first_line_model.get_key()[0])
        self.key = _key
        print()

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
            second_line_classification['growth'] = self.second_line_model.get_key()[second_line_classification['growth_code']]
            return second_line_classification
        
    def get_key(self): 
        return self.key