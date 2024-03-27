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
assuming that the filenames are in sequential order (and images were taken in the correct sequence, either ascending or
descending concentrations).
