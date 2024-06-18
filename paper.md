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

Minimum inhibitory concentration (MIC) is a laboratory test used to estimate  the susceptibility of a microorganism to an antibiotic. The result is used to determine whether it is possible to use that same drug to treat a patient's infection. Agar dilution is a reference method for MIC testing. However, the interpretation of agar dilution plates is time-consuming and prone to intra- and inter-operational error when read by laboratory personnel. `AIgarMIC` is a Python package for automated interpretation of agar dilution methodology. `AIgarMIC` processes laboratory images to identify bacterial growth on each position on solid agar containing different dilutions of an antimicrobial agent to generate a 3-dimensional growth matrix. The growth matrix is then used to identify the antimicrobial concentration at which microbial growth is inhibited -- defining the minimum inhibitory concentration. `AIgarMIC` can be imported for use in Python scripts, or can be run through a command-line interface. Users can customise `AIgarMIC` to their workflow with bespoke models, or use the pre-trained models provided. `AIgarMIC` automates the collection of multiple data and minimizes measurement error.

# Statement of need

Antimicrobial susceptibility testing (AST) is required to ensure timely and appropriate antimicrobial therapy worldwide. AST is also used to quantify the incidence and prevalence of antimicrobial resistance in hospitals, regions and countries. Agar dilution is a standard AST method -- it has the advantage of being relatively inexpensive, and enables high throughput. However, the implementation of agar dilution is often limited by the time required to interpret plates, a process that is also subject to significant intra- and inter-observer variability.

![High-level overview of the integration of ``AIgarMIC`` within the laboratory pathway of minimum inhibitory concentration measurement using agar dilution. ``AIgarMIC`` performs the interpretative steps of the pathway (from step 5), taking a set of agar plates with a colony-locating grid as an input, and reporting an MIC for each isolate.\label{fig:overview}](paper_images/overview.pdf)

The aim of `AIgarMIC` is to standardise and automate the interpretation of agar dilution plates. \autoref{fig:overview} illustrates the integration of ``AIgarMIC`` within the laboratory workflow for agar dilution MIC measurement. Typical users of `AIgarMIC` are likely to include: 

* Laboratories that are currently performing agar dilution MIC testing, but wish to automate and standardise the interpretation of their results,
* Laboratories that have a need for moderate--high throughput MIC testing, but do not have access to other automated assays and systems.  

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
