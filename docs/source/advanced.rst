Advanced Usage
==============

Using custom models
-------------------

Users who want to customize ``AIgarMIC`` or use advanced configurations are best setting up a custom python script and inheriting from ``AIgarMIC`` classes. For example, let's say that we wanted to predict images based on a very simple mean pixel value. Note that we will only be using this to demonstrate how ``AIgarMIC`` can be extended and customised (this model will not make good MIC predictions).

Firstly we will import the necessary classes and functions from ``AIgarMIC``:

>>> from aigarmic import Plate, PlateSet, Model
>>> from aigarmic import get_paths_from_directory, get_concentration_from_path

Next, we will define a custom model that inherits from :class:`aigarmic.Model`. To make MIC predictions, models must have a ``predict`` method that takes an ``opencv`` image as an input (in ``numpy`` array format), and returns a ``dict`` as an output. As a minimum, the ``dict`` should contain a ``growth_code`` key, whose value is an integer corresponding to growth code. Since this is a binary model, the code will be ``0`` or ``1``. It is also useful to provide an ``accuracy`` value (between 0 and 1) to indicate the confidence of the model's prediction. This can then be used by ``AIgarMIC`` to flag up quality issues with predictions. Here, we will not provide one, so ``AIgarMIC`` will assume the model is 100% accurate (accuracy=1.0).

>>> class MeanModel(Model):
...     def __init__(self, key=None):
...         super().__init__(key=key)
...     def predict(self, image):
...         return {'growth_code': int(image.mean() > 150)}

Now we will use :func:`aigarmic.get_paths_from_directory` to get the paths of images of agar plates, and use these to create a list of :class:`aigarmic.Plate` objects. The function :func:`aigarmic.get_concentration_from_path` is used to extract the concentration of the antibiotic from the image path.

>>> antibiotic = "amikacin"
>>> model = MeanModel(key=['no growth', 'growth'])
>>> paths = get_paths_from_directory("../images/antimicrobials/")
>>> plates = [Plate(antibiotic, get_concentration_from_path(path), model=model, image_path=path, n_row=8, n_col=12) for path in paths[antibiotic]]

Note also that we provided an (optional) key to ``MeanModel`` on construction, to help with the interpretation of the growth code. The next step involves annotating the images, i.e., converting from images to growth matrices. This step will utilise the linked ``MeanModel`` for each plate:

>>> for i in plates:
...     _ = i.annotate_images()

Finally, we can convert the list of plates to a :class:`aigarmic.PlateSet` object, and calculate the MICs:

>>> plate_set = PlateSet(plates_list=plates)
>>> _ = plate_set.calculate_mic(no_growth_key_items=tuple([0]))
>>> plate_set.convert_mic_matrix(mic_format='string').tolist()[0]
['32.0', '32.0', '32.0', '>64.0', '>64.0', '>64.0', '64.0', '32.0', '16.0', '16.0', '16.0', '16.0']

We now also have access to other ``AIgarMIC`` features such as :func:`aigarmic.PlateSet.generate_qc`.

Alternative approaches
----------------------

Alternatively, to train a keras convolutional neural network please see :ref:`train_modular` (allows training of binary or softmax model from annotate colony images using a fixed CNN structure described in http://dx.doi.org/10.1128/spectrum.04209-23), :func:`aigarmic.train_binary` and :func:`aigarmic.train_softmax`. A custom keras Sequential CNN can be defined for the latter two functions, but the training process is simplified, if a binary or softmax model is sufficient.