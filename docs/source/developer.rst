Developer notes
===============

The following is a list of notes for developers who want to contribute to ``AIgarMIC``. ``AIgarMIC`` is written in Python 3.9 and uses ``tensorflow`` and ``keras 2`` for modelling. Models have been tested only on ``keras 2`` therefore the project's dependencies require ``tensorflow < 2.16`` to maintain this compatibility.

Contributing
------------

We welcome contributions to ``AIgarMIC``. Please follow our [contributing guidelines](https://github.com/agerada/AIgarMIC/blob/paper/CONTRIBUTING.md).

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

Test coverage
-------------

`pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ is recommended for measuring tes coverage. Checking coverage for ``AIgarMIC`` requires use of `coverage <https://coverage.readthedocs.io/en/coverage.html>`_. Since some tests require the running of subprocesses (to test the CLI), ``coverage`` must be configured to monitor subprocesses. To run a test coverage report, navigate to the project root directory and run:

.. code-block:: bash

    export COVERAGE_PROCESS_START=$(pwd)/.coveragerc
    export PYTHONPATH=$(pwd)

    coverage run -m pytest --cov=aigarmic

To generate an HTML report of the coverage, run:

.. code-block:: bash

    coverage html

.. note::
    It may be necessary to use the error ignore flag (``-i``) to ignore errors depending on local configuration.

Now open the ``htmlcov/index.html`` file in a browser to view the coverage report.

.. warning::
    Use of ``coverage combine`` is not currently supported, therefore note that command line scripts (such as ``main``) are covered separately in the report.

.. note::
    As a minimum, >60% test coverage in the core parts of ``AIgarMIC`` is required for a contribution. The core parts are in: ``file_handlers.py``, ``model.py``, ``plate.py``, and ``process_plate_image.py``.

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

4. Update dependencies in ``requirements.txt`` by running (suggest use ``pip-chill`` rather than ``pip freeze`` to avoid clashes in dependencies):

.. code-block:: bash

    pip-chill > requirements.txt
    pip-chill > docs/source/requirements.txt

If developing using macOS, change the following line:

``tensorflow==2.15.0``

to:

``tensorflow==2.15.0; sys_platform != 'darwin' or platform_machine != 'arm64'``

``tensorflow-macos==2.15.0; sys_platform == 'darwin' and platform_machine == 'arm64'``

This allows platform-agnostic use.

4. Update release version in ``pyproject.toml``:

.. code-block:: bash

    poetry version patch

5. Build the package by running:

.. code-block:: bash

    poetry build

6. Commit changes and push to the repository.

7. Create a new release on GitHub and upload the built package.

Check that documentation has updated correctly on readthedocs.io: https://aigarmic.readthedocs.io/en/latest/ (note that it may take a few minutes to update).
