Installation
============

We highly recommend using an environment manager to use ``AIgarMIC``, such as ``virtualenv`` (https://virtualenv.pypa.io/en/stable/) or ``conda`` (https://docs.conda.io/en/latest/). ``AIgarMIC`` was built using Python 3.9, so we recommend using this version of Python.

For example, using conda:

.. code-block:: bash

    $ conda create -n AIgarMIC python=3.9
    $ conda activate AIgarMIC

Using pip
---------

To install the latest release of ``AIgarMIC`` from PyPI, you can run (ensure that the correct environment is activated):

.. code-block:: bash

    $ pip install AIgarMIC

.. _install-source:

Install from source
-------------------

To install the latest version of ``AIgarMIC`` from source, you can clone the repository and install it using ``pip`` or ``poetry`` (https://python-poetry.org/):

.. code-block:: bash

    $ cd path_to_your_directory
    $ conda activate AIgarMIC
    $ git clone https://github.com/agerada/AIgarMIC
    $ cd AIgarMIC
    $ pip install .

or, if poetry is installed:

.. code-block:: bash

    $ cd path_to_your_directory
    $ git clone https://github.com/agerada/AIgarMIC
    $ cd AIgarMIC
    $ poetry install

Although requirements should be installed automatically when using the above methods, they are listed in the ``requirements.txt`` and ``pyproject.toml`` files. To install requirements using ``pip``:

.. code-block:: bash

    $ pip install -r requirements.txt

.. _install-assets:

Assets installation
-------------------

Although ``AIgarMIC`` can now be used to develop you own models and pipelines, an optional collection of assets (primarily images and pre-trained models) are available to help you get started. In particular, these assets are useful when learning how to use ``AIgarMIC`` or if you do not have your own data to analyse yet. Due to their relatively large file size, these assets are not included in the main package. To download these assets, visit: https://doi.org/10.17638/datacat.liverpool.ac.uk%2F2631

Within the contents, you will find a ``README.md`` that explains the contents. Extract the contents to a directory of your choice. If you are doing development work on ``AIgarMIC``, it is recommended to extract the contents to the root of the cloned repository.
