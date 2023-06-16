# Pre-Processing & Annotation Pipeline
Before a model can be trained on a new dataset, that dataset must be annotated.
In this project, there was additional pre-processing work that needed to happen
prior to annotation. This folder contains the code & documented needed to
complete pre-processing and annotation.

* [RAW to TIF](raw_to_tif) &mdash; most orthomosaics are exported as TIF images,
  however occasionally they are exported as `RAW` images. These must be
  converted into TIFs before you can proceed with tiling and annotation.
* [Tile TIFs](tile_tifs) &mdash; large orthomosaic TIF images are divided into
  the tiles that will be annotated and used for training.
* [Collect Annotation Pools]()
* [Select Data]()
* [LOST Annotation]()
* [Combine Annotation Labels](combine_annotation_labels) &mdash; annotations
  use a hierarchical labelling scheme. Annotations are combined based on the
  level of granularity chosen for model training (i.e. should the model
  differentiate between species, age, etc?).

### Next &rarr;
Once a set of data has been annotated, it can be used within the 
[training pipeline](../2_training_pipeline).