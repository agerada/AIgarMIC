Developer notes
===============

The following is a list of notes for developers who want to contribute to ``AIgarMIC``. ``AIgarMIC`` is written in Python 3.9 and uses ``tensorflow`` and ``keras 2`` for modelling. Models have been tested only on ``keras 2`` therefore the project's dependencies require ``tensorflow < 2.16`` to maintain this compatibility.


Installation
------------

Follow instructions to clone and install ``AIgarMIC`` from source, see: :ref:`install-source`. Additionally, it is highly recommended that the optional assets are also downloaded and extracted to the project root directory, see: :ref:`install-assets`. These assets are required to run the full suite of tests.

Running tests
-------------

``AIgarMIC`` uses ``pytest`` for its testing framework. To run the full suite of tests, navigate to the project root directory and run:

.. code-block:: bash

    pytest

The suite requires the optional assets to be present in the root directory. Tests that require the assets are marked with ``@pytest.mark.assets_required``, therefore, it is possible to run a limited set of tests that do not require the assets by running:

.. code-block:: bash

    pytest -m "not assets_required"

Although this will not test the full functionality of ``AIgarMIC``, it can be useful for quickly testing that the installation succeeded.

Build process
-------------

To build a release of ``AIgarMIC``, we recommend using ``poetry``: https://python-poetry.org/. The following general steps should be followed:

1. Check pep8 compliance by running:

.. code-block:: bash

    pylint -d=R,C src/aigarmic

Errors related to ``cv2`` and ``tensorflow`` import can be ignored.

2. Run tests by running (including the optional assets):

.. code-block:: bash

    pytest

3. Test and build documentation by running:

.. code-block:: bash

    cd docs
    make doctest
    make html

4. Build the package by running:

.. code-block:: bash

    poetry build

5. Update release version in ``pyproject.toml``:

.. code-block:: bash

    poetry version patch

6. Commit changes and push to the repository.

7. Create a new release on GitHub and upload the built package.

Check that documentation has updated correctly on readthedocs.io: https://aigarmic.readthedocs.io/en/latest/ (note that it may take a few minutes to update).
