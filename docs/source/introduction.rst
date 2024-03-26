Introduction
=============

``AIgarMIC`` is a python package designed to assist bacteriologists in automating the analysis of agar dilution minimum
inhibitory concentration testing. At a high level, ``AIgarMIC`` is designed to take a set of images of agar plates, each
containing multiple strains of bacteria exposed to a different antimicrobial concentration per plate, and return a
minimum inhibitory concentration for each strain.

Functionality
=============

``AIgarMIC`` offers the following core functionality that allows users to convert a set of agar plate images to a list
of minimum inhibitory concentrations:

#. Sectioning of images into smaller images, each containing a bacterial colony (or lack thereof),
#. Classification of each colony image into a growth quality category (e.g., no growth, inhibited growth, or good growth), using a pre-trained convolutional neural network classifier,
#. Conversion of each colony image into a three dimensional matrix, where the first two dimensions represent the row and column positions of each colony inoculation, and the third dimension is the antimicrobial concentration,
#. Algorithmic determination of the minimum inhibitory concentration for each strain, using the three-dimensional growth matrix,
#. Determination of quality assurance metrics for each strain (e.g., presence of growth in the antimicrobial-negative control agar plate).

User interface
==============

Users have two options for interacting with ``AIgarMIC``:

#. Use as a traditional python package, where users can import the package, use classes, and call functions directly, or
#. Use as a command line interface application, without needing to write python code.

In general, the first option is required if users want to customize ``AIgarMIC`` or use advanced configurations. The
second option is recommended for users who want to quickly analyze a set of agar plate images using pre-trained models
and default configurations.

