---
title: 'AIgarMIC: a Python package for automated interpretation of agar dilution minimum inhibitory concentration testing'
tags:
  - Python
  - microbiology
  - image analysis
  - machine learning
  - minimum inhibitory concentration
  - bacteriology
  - laboratory software
authors:
  - name: Alessandro Gerada
    email: alessandro.gerada@liverpool.ac.uk
    orcid: 0000-0002-6743-4271
    affiliation: "1, 2"
  - name: Nicholas Harper
    email: nharper@liverpool.ac.uk
    orcid: 0009-0000-1705-0619
    affiliation: "1"
  - name: Alex Howard
    email: Alexander.Howard@liverpool.ac.uk
    orcid: 0000-0002-4195-6821
    affiliation: "1, 2"
  - name: William Hope
    email: hopew@liverpool.ac.uk
    affiliation: "1, 2"
    orcid: 0000-0001-6187-878X
affiliations:
  - name: Antimicrobial Pharmacodynamics and Therapeutics Group, Department of Pharmacology and Therapeutics, Institute of Systems, Molecular & Integrative Biology, University of Liverpool, UK
    index: 1
  - name: Department of Infection and Immunity, Liverpool Clinical Laboratories, Liverpool University Hospitals NHS Foundation Trust, Liverpool, UK
    index: 2
date: 02 April 2024
bibliography: paper.bib
---

# Summary

Minimum inhibitory concentration (MIC) is a laboratory test used to estimate  the susceptibility of a microorganism to an antibiotic. The result is used to determine whether it is possible to use that same drug to treat a patient's infection. Agar dilution is a reference method for MIC testing. However, the interpretation of agar dilution plates is time-consuming and prone to intra- and inter-operational error when read by laboratory personnel. `AIgarMIC` is a Python package for automated interpretation of agar dilution images. 

![High-level overview of the integration of `AIgarMIC` within the laboratory pathway of minimum inhibitory concentration measurement using agar dilution. `AIgarMIC` performs the interpretative steps of the pathway (from step 5), taking a set of agar plates with a colony-locating grid as an input, and reporting an MIC for each isolate. In this example, 4x4 strains are inoculated onto agar plates, giving a total of 36 strains.\label{fig:overview}](paper_images/overview.pdf)

From an input of agar plate images generated through agar dilution (usually consisting of a positive control plate and multiple plates with serial dilutions of antimicrobial concentration), `AIgarMIC` returns an MIC for each microorganism strain in the experiment. \autoref{fig:overview} provides a high-level overview of how `AIgarMIC` achieves this. Firstly, each agar plate image is split into smaller images for each bacterial strain. Next, using a pre-trained image classification model, the small colony images are converted to a code representing growth level (e.g., good growth, inhibited growth) and stored in a matrix for each plate. Finally, `AIgarMIC` uses the growth matrices from all plates to identify the antimicrobial concentration at which microbial growth is inhibited -- the minimum inhibitory concentration. `AIgarMIC` can be imported for use in Python scripts, or can be run through a command-line interface. Users can customise `AIgarMIC` to their workflow with bespoke models, or use the pre-trained models provided. `AIgarMIC` automates the collection of multiple data and minimizes measurement error.

# Software design

`AIgarMIC` can be used through a collection of [command-line scripts](https://aigarmic.readthedocs.io/en/latest/command_line_interface.html); knowledge of Python scripting is not necessary. `AIgarMIC` is designed to be extensible through a traditional Python package. The core functionality is provided through the [`Plate`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/plate/index.html#aigarmic.plate.Plate) and [`PlateSet`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/plate/index.html#aigarmic.plate.PlateSet) classes (see \autoref{fig:api_plate} for user-interface API). A [`PlateSet`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/plate/index.html#aigarmic.plate.PlateSet) instance is in essence a collection of [`Plate`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/plate/index.html#aigarmic.plate.Plate) instances, where each [`Plate`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/plate/index.html#aigarmic.plate.Plate) corresponds to an agar plate with a particular antimicrobial concentration. A minimal example is shown below, consisting of 4 strains tested over 3 dilutions/plates (+ one positive control plate):

```python
from aigarmic.plate import Plate, PlateSet
antibiotic = ["ciprofloxacin"] * 4
plate_concentrations = [0, 0.125, 0.25, 0.5]
plate_growth_matrices = []
plate_growth_matrices.append([[1, 1],
                              [0, 1]])
plate_growth_matrices.append([[1, 1],
                              [0, 0]])
plate_growth_matrices.append([[1, 0],
                              [0, 0]])
plate_growth_matrices.append([[1, 0],
                              [0, 0]])
plates = []
for ab, conc, growth in zip(antibiotic, plate_concentrations, plate_growth_matrices):
    plates.append(Plate(drug=ab, concentration=conc, growth_code_matrix=growth))
plate_set = PlateSet(plates_list=plates)
plate_set.calculate_mic(no_growth_key_items = tuple([0]))

# convert to traditional MIC values:
plate_set.convert_mic_matrix(mic_format='string').tolist()
# [['>0.5', '0.25'], ['<0.125', '<0.125']]

# check QC:
plate_set.generate_qc().tolist()
# [['P', 'P'], ['F', 'P']]
```

![`AIgarMIC` Plate and PlateSet API.\label{fig:api_plate}](paper_images/api_plate.png)

In this example, images were not used -- growth codes were provided directly in matrix format. By providing images (imported using the `opencv` library [@itseez2015opencv]) to [`Plate`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/plate/index.html#aigarmic.plate.Plate) instances, `AIgarMIC` can automatically classify growth codes using a pre-trained model. `AIgarMIC` comes with a collection of [assets](https://datacat.liverpool.ac.uk/2631/) (example images and pre-trained models) to help
users get started with the software [@geradaImageModelAssets2024]. Details of the built-in models, which are implemented as `keras` models (convolutional neural networks), can be found in the accompanying laboratory validation manuscript [@geradaDeterminationMinimumInhibitory2024].

Alternatively, users can provide a custom model by inheriting from the base [`Model`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/model/index.html#aigarmic.model.Model) class (or the [`KerasModel`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/model/index.html#aigarmic.model.KerasModel) class if using a `keras` model). Custom models must implement the `predict` method, which takes a colony image as input, and returns a `dictionary` containing, at a minimum, a `growth_code` member. \autoref{fig:api_model} shows the API for the [`Model`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/model/index.html#aigarmic.model.Model) class and subclasses. 

Further examples and tutorials can be found in the [documentation](https://aigarmic.readthedocs.io/en/latest/).

![`AIgarMIC` Model API.\label{fig:api_model}](paper_images/api_model.png)

# Statement of need

Antimicrobial susceptibility testing (AST) is required to ensure timely and appropriate antimicrobial therapy worldwide. AST is also used to quantify the incidence and prevalence of antimicrobial resistance in hospitals, regions and countries. Agar dilution is a standard AST method -- it has the advantage of being relatively inexpensive, and enables high throughput. However, the implementation of agar dilution is often limited by the time required to interpret plates, a process that is also subject to significant intra- and inter-observer variability.

The aim of `AIgarMIC` is to standardise and automate the interpretation of agar dilution plates. \autoref{fig:overview} illustrates the integration of `AIgarMIC` within the laboratory workflow for agar dilution MIC measurement. Typical users of `AIgarMIC` are likely to include: 

* Laboratories that are currently performing agar dilution MIC testing, but wish to automate and standardise the interpretation of their results,
* Laboratories that have a need for moderate--high throughput MIC testing, but do not have access to other automated assays and systems.  

# Related resources

Users of `AIgarMIC` may also be interested in the following related resources and software:

* Laboratory protocols for agar dilution MIC testing, such as those published by the European Committee on Antimicrobial Susceptibility Testing (EUCAST) [@eucastDeterminationMinimumInhibitory2000] or by Wiegand et al [@wiegandAgarBrothDilution2008]. 
* Software such as [`cellprofiler`](https://cellprofiler.org/) as a general biological image analysis tool that can be used for tasks beyond the scope of `AIgarMIC` [@lamprechtCellProfilerFreeVersatile2007].

# Laboratory validation

`AIgarMIC` has undergone research validation against a wide range of antimicrobials, against a gold standard of manual annotation. It has mainly been tested on clinical _Escherichia coli_ strains [@geradaDeterminationMinimumInhibitory2024].

# Funding and acknowledgements

`AIgarMIC` was funded, in part, by UKRI Doctoral Training Program (AG) [grant ref: 2599501] and the Wellcome Trust [grant ref: 226691/Z/22/Z]. The bacterial strains used in this study were provided by Liverpool Clinical Laboratories. 

# References
