Command-line Interface
======================

``AIgarMIC`` contains a collection of command-line scripts to help construct workflows without requiring custom python
scripts. Generally, these scripts are intended for users that are working with 96-position square agar plates. These
scripts provide the following high-level functionality:

#. :ref:`rename_images` can be used to rename a sequence of agar plate images with the correct antibiotic concentration in the filename, to prepare for analysis.
#. The main command line interface for ``AIgarMIC`` can be used to make MIC predictions using pre-trained models: :ref:`main_command_line_interface`.
#. :ref:`manual_annotator` can be used to generate a labelled dataset of bacterial colony image to train and test machine learning models.
#. In the creation of a labelled dataset, sometimes duplicate images end up in the annotation folders. :ref:`clean_up_annotations` is useful to clean up duplicate images to avoid incorrect model training.
#. The annotated datasets can be used to train a binary or softmax model using :ref:`train_modular`. This creates a model that can be used as an input to :ref:`main_command_line_interface`.
#. Finally, :ref:`model_performance` is useful to test the model's performance on an unseen dataset that is held for testing purposes.

.. _rename_images:

Rename images
-------------

.. argparse::
   :module: aigarmic.rename_images
   :func: rename_images_parser
   :prog: rename_images

.. _main_command_line_interface:

``AIgarMIC`` -- main command-line interface
-------------------------------------------

.. argparse::
   :module: aigarmic.main
   :func: main_parser
   :prog: main

.. _manual_annotator:

Manual annotator
----------------

.. argparse::
   :module: aigarmic.manual_annotator
   :func: manual_annotator_parser
   :prog: manual_annotator

.. _clean_up_annotations:

Clean up annotations
--------------------

.. argparse::
   :module: aigarmic.clean_up_annotations
   :func: clean_up_annotations_parser
   :prog: clean_up_annotations

.. _train_modular:

Modular model training script
-----------------------------

.. argparse::
   :module: aigarmic.train_modular
   :func: train_modular_parser
   :prog: train_modular

.. _model_performance:

Model performance
-----------------

.. argparse::
   :module: aigarmic.model_performance
   :func: model_performance_parser
   :prog: model_performance
