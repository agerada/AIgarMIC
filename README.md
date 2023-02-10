# Agar Dilution MIC image reader

## Introduction

Set of scripts for automated annotation and grading of agar dilution plates to facilitate high throughput homebrew MIC testing. 

## Requirements

Currently the scripts have the following requirements (but extension is possible to other scenarios): 

* 96 inoculations per agar plate
* splitting of sub-images is based on a black grid background 
* absence of other black colors on the plates, which would interfere with the threshold based automated image splitting

The following image is an ideal example: 
![Example image 1](/images/example_plates/0.5.jpg)

Some shearing is allowed and corrected in the pre-processing: 
![Example image 2](/images/example_plates/128.jpg)

## Usage 

### Install 

Install dependencies using `requirements.txt`. 

### Demo

To run a demonstration of the image processing module: 

    python demo.py

### Manual Annotation

In order to generate manually annotated images, use (IMP: -o MUST be used): 

    usage: manual_annotator.py [-h] [-o] directory

    Manually annotate plate images

    positional arguments:
        directory             Directory containing plate images

    optional arguments:
      -h, --help            show this help message and exit
      -o, --output_to_files
            Output to .jpg files in subfolders in annotations/ If not used, then defaults to storing in .p pickle files (NOT IMPLEMENTED) [ default FALSE ]

The following scoring codes are used throughout the project: 

* `0`: no growth
* `1`: faint film of growth or isolated colony
* `2`: good growth

### Model training

Use tensorflow to train image model. This script uses annotated images from `annotations/`. Suggest storing models in `models/` for ease of use in `main.py`. 

    python train_model.py 

    usage: 
        This script loads images from annotations directory and trains ML model. Loading from pickled data is not yet implemented
    
       [-h] [-p] [-v] [-s SAVE]

    optional arguments:
        -h, --help            show this help message and exit
        -p, --pickled         Load data from pickled (.p) files - NOT IMPLEMENTED
        -v, --visualise       Generate visualisations for model diagnostics
        -s SAVE, --save SAVE  If specified, tensorflow model will be saved to this folder

### Annotate and intepret agar plate

Use `main.py`. At present this script reads example plates for gentamicin and shows image annotations using tensorflow model. 

## To-Do

* Algorithm to calculate MIC from annotations
* QC (controls and "skipped" plates)
* Option to save output
