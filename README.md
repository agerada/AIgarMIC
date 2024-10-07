# `AIgarMIC`

[![DOI](https://joss.theoj.org/papers/10.21105/joss.06826/status.svg)](https://doi.org/10.21105/joss.06826)

## Introduction

`AIgarMIC` is a Python package and collection of commandline scripts designed to facilitate the automation of agar dilution minimum inhibitory concentration image interpretation.

`AIgarMIC` has the following features:

* Automated image processing of agar dilution plates in the following format (note the use of an anchoring black grid to delineate colonies): 

![Example image 1](/readme_images/2.0.jpg)

* Flexible MIC calculation algorithm with ability to disregard inhibited growth
* Quality assurance metrics to ensure MIC predictions
* Pre-trained models and example datasets
* Scripts to support custom model training

## Documentation

The full documentation for `AIgarMIC` can be found at:

https://aigarmic.readthedocs.io/en/latest/

## Installation

To install `AIgarMIC`, follow the instructions below:

https://aigarmic.readthedocs.io/en/latest/installation.html

## Usage

To use `AIgarMIC`, follow one of the typical workflows described below:

https://aigarmic.readthedocs.io/en/latest/introduction.html#typical-workflows

## Author information

The lead developer of ``AIgarMIC`` is Alessandro Gerada (https://github.com/agerada/ and https://agerada.github.io/), 
University of Liverpool, UK (alessandro.gerada@liverpool.ac.uk).

## Cite

If you are using `AIgarMIC` in your research project, please cite [TO FOLLOW].

To cite the validation data and developmental approach described in the `AIgarMIC` validation manuscript, please cite:

    @article{geradaDeterminationMinimumInhibitory2024,
      title = {Determination of Minimum Inhibitory Concentrations Using Machine-Learning-Assisted Agar Dilution},
      author = {Gerada, Alessandro and Harper, Nicholas and Howard, Alex and Reza, Nada and Hope, William},
      editor = {Shier, Kileen L.},
      date = {2024-03-22},
      journaltitle = {Microbiology Spectrum},
      shortjournal = {Microbiol Spectr},
      pages = {e04209-23},
      issn = {2165-0497},
      doi = {10.1128/spectrum.04209-23},
      url = {https://journals.asm.org/doi/10.1128/spectrum.04209-23},
      urldate = {2024-04-02},
      langid = {english}
    }

## External links

The manuscript describing the validation of `AIgarMIC` can be found at: https://doi.org/10.1128/spectrum.04209-23.
Optional asset data is available at: https://doi.org/10.17638/datacat.liverpool.ac.uk%2F2631.

## Contributing

We welcome contributions to ``AIgarMIC``. Please follow our [contributing guidelines](https://github.com/agerada/AIgarMIC/blob/paper/CONTRIBUTING.md).

## License

`AIgarMIC` is provided under the GNU General Public License v3.0. For more information, see the LICENSE file.
