# Count Comparisons with Plots
> Note: This task could be moved inside the manuscript_reporting directory

Once the model has made predictions & those predictions have been run through
the post-processing pipeline, we can plot the results of those predictions.

## RMSE Plots
Root mean squared error (RMSE) plots are used to select confidence thresholds
for each class. These thresholds are selected based off a set of "validation"
panoramas, for which manual counts are available. These panoramas _should not_ 
have been used to train models nor to inform the selection of hyperparameters. 

Five arguments need to be provided when creating the RMSE plots &mdash; 
`--plot_type`, `--true_counts`, `--detections_dir`, `--file_map`, and
`--out_path` (each of which are described in detail [below](#arg-detail)).
Additionally, the `--save_raw_csv` flag can be used, but is optional (also
described below). The following code block shows an example of how to create
RMSE plots.
```commandline
python3 compare_counts.py \
--plot_type rmse \
--true_counts ../input/SNB_2020_VAL/manual_counts_verified.csv \
--detections_dir ../input/SNB_2020_VAL/snb5_hg_detections \
--file_map ../input/SNB_2020_VAL/snb5_file_map.json \
--out_path ../output/SNB_2020_VAL/rmse.png
```

<a name='arg-detail'></a>
### `--plot_type`
The `--plot_type` argument is used to specify what kind of plot to plot. There
are two options (`rmse` and `count`). To plot an RMSE plot, you should use
`--plot_type=rmse`. :exploding_head:
### `--true_counts`
`--true_counts` should be provided with the filepath to the CSV containing the 
manual counts you want to compare with your model results. There should be three
columns in this file:   
* `Date` which should be formatted as 
`YYYY-MM-DD`
* `Cormorant` which contains the number of cormorants
* `Nest` which contains the number of nests    

The code block below contains an example CSV.
```csv
Date,Cormorant,Nest
2020-06-08,242,150
2020-06-17,232,148
```
### `--detections_dir`
`--detections_dir` should be provided the path to the directory containing the
detection files you want to plot. The code block below shows an example
structure of this directory (`snb5-cn-hg-detections`) which has been placed
inside the `input` directory.
```commandline
input
└───snb5-cn-hg-detections
    │   model_20200608.csv
    │   model_20200617.csv
    │   ...
    │   model_mask_20200608.csv
    │   model_mask_20200617.csv
    │   ...
    │   model_mask_dedup_20200608.csv
    │   model_mask_dedup_20200617.csv
    └───...
```
### `--file_map`
`--file_map` should be provided with the filepath to a JSON file that specifies
the date & method for each detection file contained within `--detections_dir`.
The format of this file should be as follows: 
```json
{
  "Method 1": {
    "YYYY-MM-DD": "detection_file.csv",
    "YYYY-MM-DD": "detection_file.csv",
  },
  
  "Method 2": {
    "YYYY-MM-DD": "detection_file.csv",
    "YYYY-MM-DD": "detection_file.csv",
  }
}
```
See the code block below for a completed example of how this JSON file might
look.
```json
{
  "Model": {
    "2020-06-08": "model_20200608.csv",
    "2020-06-17": "model_20200617.csv"
  },
  
  "Model + Masking": {
    "2020-06-08": "model_mask_20200608.csv",
    "2020-06-17": "model_mask_20200617.csv"
  },
  
    "Model + Masking + Nest Deduplication": {
    "2020-06-08": "model_mask_dedup_20200608.csv",
    "2020-06-17": "model_mask_dedup_20200617.csv"
  }
}
```

### `--out_path`
The `--out_path` argument is used to specify the filepath where the generated
plot should be saved. For example, 
`--output_path=../output/SNB_2020_VAL/rmse.png`. 

### `--save_raw_csv`
The optional, `--save_raw_csv` flag determines whether the raw data used to
generate the plot are saved to a CSV. If this flag is used, a CSV containing the
RMSEs for each confidence score threshold and label pair will be saved in
the parent directory of the filepath specified in the `--output_path` argument.
This file will be named `rmse.csv`.  

## Count Plots

## The `create_plots.sh` Script
