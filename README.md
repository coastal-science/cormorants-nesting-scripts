# cormorants-nesting-scripts
This repository contains all the Python code relevant for the Cormorants BC Nesting project. 

The top-level directory contains the following folders: 
1. [`gigapan_data_inventory`](/gigapan_data_inventory) &mdash; Contains a text file detailing what data we have for this project (last updated Feb 10, 2022). This includes reference to which data were used for training, validating, and testing the CNN models. 
2. [`object_detection_scripts`](/object_detection_scripts) &mdash; Contains Python code for the object detection pipeline. Within this directory, you'll find many more subdirectories, each corresponding to a particular task in the pipeline. Each subdirectory is composed of a `input`, `output`, and `src` folder. The structure of these directories is described in more detail within the `object_detection_scripts` [README.md](/object_detection_scripts/README.md). There you will also find instructions for running the Python code (on either your local machine or on a cluster system). 
3. [`docs`](/docs) &mdash; Contains documentation for working with trained models, Cedar, and the commandline generally.