Quickstart guide
================

Aim
---

This is a short introduction to ``AIgarMIC``. It is intended to give a brief overview of the main features so that most
typical users can quickly start calculating minimum inhibitory concentrations (MICs) of bacterial strains against
antimicrobials.

Data prerequisites
------------------

Grid
^^^^

The input data for ``AIgarMIC`` is a set of images containing multiple bacterial strains, each image corresponding to
a different antimicrobial concentrations. Multiple antimicrobials can be calculated. The key requirement for images is
that there is a black grid overlaying (or underlying) the image, with each square surrounding a bacterial inoculation
position. The grid is used by the software to identify the positions of each bacterial strains.

``AIgarMIC`` does not itself apply the grid, since there is no single fixed position. There are different ways that the
grid could be applied. While automated approaches e.g., using corner colonies as anchors, are initially appealing, from
our experience it is best to apply the grid manually in a software such as ``GIMP``: https://www.gimp.org/.
This ensures that the grid is aligned correctly, and that ``AIgarMIC`` can accurately identify the grid.
Other alternatives include applying the grid at imaging stage using a camera with a grid overlay, or using a grid
printed on transparent film. An example of a 96-well grid is provided in the optional assets dataset: :ref:`install-assets`.
Examples of agar plate images with grids applied are also provided.

Folder structure
^^^^^^^^^^^^^^^^

Since we will be using ``AIgarMIC`` in command line mode for this example, we need to have images organised in a
particular folder structure. The folder structure should be as follows::


    .
    ├── images
    │   ├── antimicrobial1
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── antimicrobial2
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── ...
    └── ...


Where ``images`` is the root folder containing all the images, and each subfolder corresponds to a different antimicrobial.
Again, an example of this folder structure is provided in the optional assets dataset: :ref:`install-assets` (see ``images/antimicrobials/``).

Filenames
^^^^^^^^^

The images should be named in a way that allows ``AIgarMIC`` to identify the antimicrobial concentration. This is done
by simply naming each antimicrobial concentration in the filename, e.g.,::

    0.jpg
    0.5.jpg
    1.jpg
    2.jpg

Note that ``0.jpg`` is the control image, with no antimicrobial.

To facilitate image renaming, ``AIgarMIC`` provides a script that can be used to rename images in the correct format,
assuming that the filenames are in lexical and sequential order (and images were taken in the correct sequence, either
ascending or descending concentrations). See :ref:`rename_images`.

Model/s
^^^^^^^

To calculate MICs, ``AIgarMIC`` needs to determine whether there is growth at every position in each agar plate. To do,
this we will provide a pre-trained model to make these predictions. We will use the models provided in the optional
assets dataset, available in ``models/spectrum_2024/``. There, we will find two models:

    `growth_no_growth/`
    `good_growth_poor_growth/`

These are pre-trained `keras` models described in the validation manuscript: http://dx.doi.org/10.1128/spectrum.04209-23
The models are convolutional neural networks which predict whether a colony image has growth and the quality of growth
respectively. The models are therefore binary classifiers. ``AIgarMIC`` can use the models in sequence (first-step and
second-step) to determine growth.

Installation
------------

To install ``AIgarMIC`` and its dependencies, please refer to the installation guide: :doc:`installation`.

Running `AIgarMIC`
------------------

Firstly, activate the `AIgarMIC` virtual environment that was created during installation. For example, for a `conda`
environment, this can be done by running::

    conda activate AIgarMIC

Then, run the following command to calculate MICs for the provided example dataset:

.. code-block:: bash

    cd path/to/optional_assets
    AIgarMIC -m models/spectrum_2024/growth_no_growth/ models/spectrum_2024/good_growth_poor_growth/ -t binary -n 0,1 -r 160 160 -d 8 12 -o output/results.csv images/antimicrobials/amikacin

Where,

        - ``-m``: the path to the models to be used. We are using a two-step model, therefore provide two paths.
        - ``-t``: the type of model to be used. In this case, we are using binary models.
        - ``-n``: the growth codes that should be counted as negative growth. The models described in the manuscript annotate images with the following codes: 0 (no growth), 1 (good growth), 2 (poor growth). In this case, we are considering no growth (0) and good growth (1) as negative growth.
        - ``-r``: the resolution of the colony images. In this case, the images are 160x160 pixels (the default).
        - ``-d``: the dimensions of the agar grid. In this case, the grid is 8x12 (the default), giving 96 total colonies per agar plate image.
        - ``-o``: the output file where the results will be saved.
        - ``images/antimicrobials/amikacin``: the path to the images to be analysed (the only positional argument)

`AIgarMIC` supports multiple antimicrobials, under the same folder structure. To get help, use:

.. code-block:: bash

    AIgarMIC -h

Conclusion
----------

On inspection of ``output/results.csv``, we find MICs for each strain position. Results are provided with plate
positions that correspond to a 96-well micro-plate. This script currently only supports 96-position inoculations.
In addition to MICs, we also get a quality assurance score (QC) for each MIC, where:

    - ``'P'``: PASS -- no anomalies detected,
    - ``'F'``: FAIL -- no growth in the negative control plate that does not have antimicrobial (note that the negative control position `should` FAIL),
    - ``'W'``: WARNING -- changes in growth patterns is not as expected; generally this means that plates have the following pattern: growth -> no growth -> growth (as concentrations increase); results should be checked to confirm.

