Command-line Interface
======================

``AIgarMIC`` -- main command-line interface
-------------------------------------------

.. argparse::
   :filename: ../src/aigarmic/main.py
   :func: main_parser
   :prog: main

.. _clean_up_annotations:

Clean up annotations
--------------------

.. argparse::
   :module: aigarmic.clean_up_annotations
   :func: clean_up_annotations_parser
   :prog: clean_up_annotations

.. _manual_annotator:

Manual annotator
----------------

.. argparse::
   :module: aigarmic.manual_annotator
   :func: manual_annotator_parser
   :prog: manual_annotator

.. _model_performance:

Model performance
-----------------

.. argparse::
   :module: aigarmic.model_performance
   :func: model_performance_parser
   :prog: model_performance

.. _rename_images:

Rename images
-------------

.. argparse::
   :module: aigarmic.rename_images
   :func: rename_images_parser
   :prog: rename_images

.. _train_modular:

Modular model training script
-----------------------------

.. argparse::
   :module: aigarmic.train_modular
   :func: train_modular_parser
   :prog: train_modular
