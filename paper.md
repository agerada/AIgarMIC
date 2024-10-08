---
title: 'AIgarMIC: a Python package for automated interpretation of agar dilution minimum inhibitory concentration assays'
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
  - name: Antimicrobial Pharmacodynamics and Therapeutics Group, Department of Pharmacology and Therapeutics, Institute of Systems, Molecular & Integrative Biology, University of Liverpool, United Kingdom
    index: 1
  - name: Department of Infection and Immunity, Liverpool Clinical Laboratories, Liverpool University Hospitals NHS Foundation Trust, Liverpool, United Kingdom
    index: 2
date: 02 April 2024
bibliography: paper.bib
---

# Summary

Minimum inhibitory concentration (MIC) assays are used to estimate the susceptibility of a microorganism to an antibiotic. The result is broadly used within microbiology. In clinical settings, it is used to determine whether it is possible to use that same drug to treat a patient's infection. Agar dilution is a reference method for MIC measurement. However, the interpretation of agar dilution plates is time-consuming and prone to intra- and inter-operational errors when read by laboratory personnel. `AIgarMIC` is a Python package for automated interpretation of agar dilution images. 

![High-level overview of the integration of `AIgarMIC` within the laboratory pathway of minimum inhibitory concentration measurement using agar dilution. `AIgarMIC` performs the interpretative steps of the pathway (from step 5), taking a set of agar plates with a colony-locating grid as an input, and reporting an MIC for each isolate. In this example, 4x4 strains are inoculated onto agar plates, giving a total of 16 strains. F = quality control failed (no growth in positive control plate).\label{fig:overview}](paper_images/overview.pdf)

From an input of agar plate images generated through agar dilution (usually consisting of a positive control plate and multiple plates with serial dilutions of antimicrobial concentration), `AIgarMIC` returns an MIC for each microorganism strain in the experiment. \autoref{fig:overview} provides a high-level overview of how `AIgarMIC` achieves this. Firstly, each agar plate image is split into smaller images for each bacterial strain. Next, using a pre-trained image classification model, the small colony images are converted to a code representing growth level (e.g., good growth, inhibited growth) and stored in a matrix for each plate. Finally, `AIgarMIC` uses the growth matrices from all plates to identify the antimicrobial concentration at which microbial growth is inhibited -- the minimum inhibitory concentration. `AIgarMIC` can be imported for use in Python scripts, or can be run through a command-line interface. Users can customise `AIgarMIC` to their workflow with bespoke models, or use the pre-trained models provided. `AIgarMIC` automates the collection of multiple data, reduces human error, and reduces subjective operator variability.

# Software design

`AIgarMIC` can be used through a collection of [command-line scripts](https://aigarmic.readthedocs.io/en/latest/command_line_interface.html); knowledge of Python scripting is not necessary. Given a collection of images from one or more agar dilution experiments, `AIgarMIC` can calculate the MIC from a single script:

```bash
    AIgarMIC -m model/ -t binary -n 0 -d 8 12 -r 160 160 -o results.csv images/
```

Where, 

- `-m`, `--model` specifies the path to the pre-trained model,
- `-t`, `--type_model` specifies the type of model (binary or softmax),
- `-n`, `--negative_codes` specifies the growth code/s that should be classed as no growth,
- `-d`, `-dimensions` specifies the dimensions of the agar plate images (8 rows and 12 columns in this example),
- `-r`, `--resolution` specifies the resolution of the images (x and y) on which the model was trained,
- `-o`, `--output_file` specifies the target `.csv` output file

`AIgarMIC` is designed to be extensible through a Python package. The core functionality is provided through the [`Plate`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/plate/index.html#aigarmic.plate.Plate) and [`PlateSet`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/plate/index.html#aigarmic.plate.PlateSet) classes (see \autoref{fig:api_plate} for user-interface API). A [`PlateSet`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/plate/index.html#aigarmic.plate.PlateSet) instance is in essence a collection of [`Plate`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/plate/index.html#aigarmic.plate.Plate) instances, where each [`Plate`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/plate/index.html#aigarmic.plate.Plate) corresponds to an agar plate with a particular antimicrobial concentration. A minimal example is shown below, consisting of 4 strains tested over 3 dilutions/plates (+ one positive control plate):

```python
from aigarmic.plate import Plate, PlateSet

# set up 4 plates of ciprofloxacin (including a positive control plate)
antibiotic = ["ciprofloxacin"] * 4
plate_concentrations = [0, 0.125, 0.25, 0.5]

# temporary list of growth codes for each plate
# each plate has 2x2 inoculated strains
plate_growth_matrices = []
plate_growth_matrices.append([[1, 1],
                              [0, 1]])
plate_growth_matrices.append([[1, 1],
                              [0, 0]])
plate_growth_matrices.append([[1, 0],
                              [0, 0]])
plate_growth_matrices.append([[1, 0],
                              [0, 0]])

# combine data into Plate instances
plates = []
for ab, conc, growth in zip(antibiotic,
                            plate_concentrations,
                            plate_growth_matrices):
    plates.append(Plate(drug=ab,
                        concentration=conc,
                        growth_code_matrix=growth))

# create PlateSet instance using list of Plates
plate_set = PlateSet(plates_list=plates)

plate_set.calculate_mic(
    no_growth_key_items = tuple([0])) # growth codes that indicate no growth

plate_set.mic_matrix.tolist()
# [[1.0, 0.25], [0.125, 0.125]]

# convert to traditional MIC values:
plate_set.convert_mic_matrix(mic_format='string').tolist()
# [['>0.5', '0.25'], ['<0.125', '<0.125']]

# check QC:
plate_set.generate_qc().tolist()
# [['P', 'P'], ['F', 'P']]
```

![`AIgarMIC` Plate and PlateSet API.\label{fig:api_plate}](paper_images/api_plate.png)

In this example, images were not used -- growth codes were provided directly in matrix format. By providing images (imported using the [`opencv`](https://github.com/itseez/opencv) library to [`Plate`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/plate/index.html#aigarmic.plate.Plate) instances, `AIgarMIC` can automatically classify growth codes using a pre-trained model. `AIgarMIC` comes with a collection of [assets](https://datacat.liverpool.ac.uk/2631/) (example images and pre-trained models) to help
users get started with the software [@geradaImageModelAssets2024]. Details of the built-in models, which are implemented as [`keras`](https://keras.io/) models (convolutional neural networks), can be found in the accompanying laboratory validation manuscript [@geradaDeterminationMinimumInhibitory2024].

Alternatively, users can provide a custom model by inheriting from the base [`Model`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/model/index.html#aigarmic.model.Model) class (or the [`KerasModel`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/model/index.html#aigarmic.model.KerasModel) class if using a [`keras`](https://keras.io/) model). Custom models must implement the `predict` method, which takes a colony image as input, and returns a `dictionary` containing, at a minimum, a `growth_code` member. \autoref{fig:api_model} shows the API for the [`Model`](https://aigarmic.readthedocs.io/en/latest/autoapi/aigarmic/model/index.html#aigarmic.model.Model) class and subclasses. 

To support users in developing custom models, `AIgarMIC` provides an [annotator script](https://aigarmic.readthedocs.io/en/latest/command_line_interface.html#manual-annotator) that allows users to generate annotated colony images to train and test a custom model. The generated labelled images can be used to train a model using a training [script](https://aigarmic.readthedocs.io/en/latest/command_line_interface.html#train-modular) (which uses the neural network architecture design reported in @geradaDeterminationMinimumInhibitory2024). Other convenience features are available within the [command line interface](https://aigarmic.readthedocs.io/en/latest/command_line_interface.html). 

Further examples and tutorials can be found in the [documentation](https://aigarmic.readthedocs.io/en/latest/).

![`AIgarMIC` Model API.\label{fig:api_model}](paper_images/api_model.png)

# Statement of need

Antimicrobial susceptibility testing (AST) is required to ensure timely and appropriate antimicrobial therapy worldwide. Susceptibility testing is also used to quantify the incidence and prevalence of antimicrobial resistance in hospitals, regions and countries. Agar dilution is a standard AST method -- it has the advantage of being relatively inexpensive and enables high throughput. However, since most agar dilution experiments are interpreted by visually inspecting agar plates of microbial growth, the implementation of agar dilution is often limited by this time-consuming step. Visual inspection is also subject to human error, and the inherent subjectivity of classifying colony growth can lead to significant intra- and inter-observer variability.

The aim of `AIgarMIC` is to standardise and automate the interpretation of agar dilution plates, reducing the impact of human error on results. Furthermore, since the performance of `AIgarMIC` is fixed to a particular model, MIC results are not subject to operator variability. Hence, MIC results can be more reliably compared between experiments and laboratories. \autoref{fig:overview} illustrates the integration of `AIgarMIC` within the laboratory workflow for agar dilution MIC measurement. Typical users of `AIgarMIC` are likely to include: 

* Laboratories that are currently performing agar dilution MIC testing, but wish to automate and standardise the interpretation of their results,
* Laboratories that need moderate--high throughput MIC testing, but do not have access to other automated assays and systems.  

# Related resources

Users of `AIgarMIC` may also be interested in the following related resources and software:

* Laboratory protocols for agar dilution MIC assays, such as those published by the European Committee on Antimicrobial Susceptibility Testing (EUCAST) [@eucastDeterminationMinimumInhibitory2000] or by Wiegand et al. [@wiegandAgarBrothDilution2008]. 
* Software such as [`cellprofiler`](https://cellprofiler.org/) as a general biological image analysis tool that can be used for tasks beyond the scope of `AIgarMIC` [@lamprechtCellProfilerFreeVersatile2007].

# Laboratory validation

`AIgarMIC` has undergone research validation against a wide range of antimicrobials, against a gold standard of manual annotation. It has mainly been tested on clinical _Escherichia coli_ strains [@geradaDeterminationMinimumInhibitory2024].

# Funding and acknowledgements

`AIgarMIC` was funded, in part, by UKRI Doctoral Training Program (AG) [grant ref: 2599501] and the Wellcome Trust [grant ref: 226691/Z/22/Z]. The bacterial strains used in this study were provided by Liverpool Clinical Laboratories. 

# References
