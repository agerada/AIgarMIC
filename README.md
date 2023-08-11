# Agar Dilution MIC image reader

## Introduction

Set of scripts for automated annotation and grading of agar dilution plates to facilitate high throughput homebrew MIC testing. 

## Requirements

Currently the scripts have the following requirements (but extension is possible to other scenarios): 

* 96 inoculations per agar plate
* splitting of sub-images is based on a black grid background 
* absence of other black colors on the plates, which would interfere with the threshold based automated image splitting

The following image is an ideal example: 
![Example image 1](/readme_images/0.5.jpeg)

Some shearing is allowed and corrected in the pre-processing: 
![Example image 2](/readme_images/128.jpeg)

## Usage 

### Install 

The recommended setup is to use the supplied conda environment (agar-diluation): 

    $ conda env create --file environment.yaml
    $ conda activate agar-dilution
    (agar-dilution) $ python demo.py

Alternatively install dependencies using pip and `requirements.txt`. 

### Prep work 

Organise your image in a directory tree that satisfies `main.py` requirements (see below). For example, 

    \date_of_run\antibiotic_name\concentration1..n.JPG

For batch rotation of images, an automated tool such as imagemagick may be useful: 

    $ for image in *.JPG; do convert $image -rotate 90 rotated-$image; done

### Demo

To run a demonstration of the image processing module: 

    $ python demo.py

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

    $ python train_model.py 

    usage: 
        This script loads images from annotations directory and trains ML model. Loading from pickled data is not yet implemented
    
       [-h] [-p] [-v] [-s SAVE]

    optional arguments:
        -h, --help            show this help message and exit
        -p, --pickled         Load data from pickled (.p) files - NOT IMPLEMENTED
        -v, --visualise       Generate visualisations for model diagnostics
        -s SAVE, --save SAVE  If specified, tensorflow model will be saved to this folder

### Annotate and intepret agar plate

Use `main.py`: 

    usage: main.py [-h] [-m MODEL] [-o OUTPUT_FILE] [-s] directory

    Main script to interpret agar dilution MICs

    positional arguments:
    directory             
                                Directory containing images to process, arranged in subfolders by antibiotic name, e.g.,: 
                                    directory/ 
                                            antibiotic1_name/ 
                                                    0.jpg 
                                                    0.125.jpg 
                                                    0.25.jpg 
                            
    optional arguments:
    -h, --help            show this help message and exit
    -m MODEL, --model MODEL
                            Specify file containing tensorflow model for image classificaion
    -o OUTPUT_FILE, --output_file OUTPUT_FILE
                            Specify output file for csv report (will be overwritten)
    -s, --suppress_validation
                            Suppress manual validation prompts for annotations that have poor accuracy

It is recommended that manual validation is not suppressed, unless running through a bash script for example. Manual validation prompts the user to confirm image annotation when the model predicts with a <90% accuracy. In addition, the new labelled images are exported to `new_annotations/` with the same structure as `annotations/`

# Plotting

`model_plot.py` has the following dependencies: 

    conda install -c conda-forge pygraphviz
    pip install graphviz
    pip install pydot

## To-Do

* ~~Algorithm to calculate MIC from annotations~~
* ~~QC (controls and "skipped" plates)~~
* ~~Option to save output~~
* ~~Parallel processing of plate image labelling~~ (unlikely to be worth effort)
* Improve model fit - in progress (method to flag up poor annotations)
* Specify control positions and expected MIC for control strain
