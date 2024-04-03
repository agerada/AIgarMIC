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
affiliations:
  - name: Antimicrobial Pharmacodynamics and Therapeutics Group, Department of Pharmacology and Therapeutics, Institute of Systems, Molecular & Integrative Biology, University of Liverpool, UK
    index: 1
  - name: Department of Infection and Immunity, Liverpool Clinical Laboratories, Liverpool University Hospitals NHS Foundation Trust, Liverpool, UK
    index: 2
date: 02 April 2024
bibliography: paper.bib
---

# Summary

The measurement of minimum inhibitory concentration (MIC) is a gold standard test for the measurement of antimicrobial
susceptibility. Although agar dilution is a reference method for MIC testing, the interpretation of agar dilution
plates is time-consuming and subjective. `AIgarMIC` is a Python package for automated interpretation of agar dilution
experiments. Starting with agar plate images, each containing a fixed antimicrobial concentration and multiple strains, `AIgarMIC` processes the images to identify colony
growth in each position and generate a 3-dimensional growth matrix. The growth matrix is then used to identify the plate
at which growth is first inhibited -- corresponding to the minimum inhibitory concentration. `AIgarMIC` can be imported
for use in Python scripts, or can be run through a command-line interface. Users can customise `AIgarMIC` to their
workflow with bespoke models, or use the pre-trained models provided.

# Statement of need

Most microbiology laboratories have a requirement to perform antimicrobial susceptibility testing (AST) using minimum
inhibitory concentration. Research applications include testing of new antimicrobial agents, while clinical applications
include the calculation of resistance rates to understand antimicrobial resistance (AMR) epidemiology. While large
laboratories may have access to automated systems, many laboratories use manual methods to perform MIC. Agar dilution
is an appealing method for MIC testing, as it is relatively inexpensive, is regarded as a reference method, and has a high testing throughput. However, the implementation of agar dilution is often limited by two factors: the skill required to generate the correct organism inoculum, and the time required to interpret
plates and calculate the MIC. The latter is also subject to inter-observer variability and human error. 

The aim of `AIgarMIC` is to standardise and automate the interpretation of agar dilution plates. Typical users of `AIgarMIC` are likely to include: 

* Laboratories who are currently performing agar dilution MIC testing, but who wish to automate and standardise the interpretation of their results,
* Laboratories who have a need for moderate--high throughput MIC testing, but who do not have access to automated 
 systems.  

# Related resources

Users of `AIgarMIC` may also be interested in the following related resources and software:

* Laboratory protocols for agar dilution MIC testing, such as those published by the European Committee on Antimicrobial Susceptibility Testing (EUCAST) [@eucastDeterminationMinimumInhibitory2000] or by Wiegand et al [@wiegandAgarBrothDilution2008]. 
* Software such as [`cellprofiler`](https://cellprofiler.org/) as a general biological image analysis tool that can be used for tasks beyond the scope of `AIgarMIC` [@lamprechtCellProfilerFreeVersatile2007].

Additionally, `AIgarMIC` also comes with a collection of [assets](https://datacat.liverpool.ac.uk/2631/) (example images and pre-trained models) to help
users get started with the software [@geradaImageModelAssets2024]. 

# Laboratory validation

`AIgarMIC` has undergone research validation against a wide range of antimicrobials, against a gold standard of manual annotation. It has mainly been tested on clinical _Escherichia coli_ strains [@geradaDeterminationMinimumInhibitory2024].

# Funding and acknowledgements

`AIgarMIC` was funded, in part, by UKRI Doctoral Training Program (AG) [grant ref: 2599501] and the Wellcome Trust [grant ref: 226691/Z/22/Z]. The bacterial strains used in this study were provided by Liverpool Clinical Laboratories. 

# References
