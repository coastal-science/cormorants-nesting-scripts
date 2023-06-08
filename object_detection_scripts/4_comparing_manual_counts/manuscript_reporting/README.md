# Manuscript Reporting


## Annotation Statistics
The `annotation_statistics.py` file contains the functionality for analyzing a
set of annotations created by manual users in the LOST annotation
web-application. 

The set of annotation files to be analyzed is specified within a JSON file, 
whose path is provided using the `--anno_list` argument. The code block below 
shows an example of how this JSON file should be formatted.
```json
[
  "../input/snb-2021/user1_20210601_annos.csv", 
  "../input/snb-2021/user2_20210608_annos.csv",
  "../input/snb-2021/user1_20210615_annos.csv",
  "../input/snb-2021/user2_20210622_annos.csv"
]
```
The `--stat` argument is used to specify what annotation statistic you are
interested in and can be one of `--stat=count`, `--stat=time`, and
`--stat=size`. The three sections below provide more detail on the metrics
provided these three `--stat` options.

### Number of Annotations
To find statistics related to the number of annotations, you need to specify
`--stat=count`. The `--level` argument is used to specify the level of
aggregation to use for calculating these numbers and has three possible values
(in increasing granularity) &mdash; `all`, `pano`, and `tile`. Using 
`--level=all` will calculate the total number of annotations contained within
the provided annotation files, . `--level=pano` and `--level=tile` will 
calculate the number of annotations per panorama and per tile, respectively. 

#### Example 1
The code block below shows an example of how to calculate the total number of 
annotations across all annotation files.
```commandline
python3 annotation_statistics.py\
--anno_list ../input/snb-5-annos.json \
--stat=count \
--level=all
```
```text
Total Number of Annotations:
Cormorant    1641
Nest         1153
```

#### Example 2
The code block below shows an example of how to calculate statistics related to
the number of annotations per tile. 
```commandline
python3 annotation_statistics.py \
--anno_list ../input/snb-5-annos.json  \
--stat=count \
--level=tile
```
```text
Annotations per Tile:
mean      7.04
std      16.97
min       0.00
25%       0.00
50%       0.00
75%       6.00
max     159.00
Name: anno.data, dtype: float64
```
### Annotation Time
To find statistics related to the amount of time that was spent on annotation
you need to specify `--stat=time`. The `--level` parameter is used to specify
the level of aggregation to use for calculations and has four possible values.
Specifying `--level=all` calculates the total annotation time for the annotation files
provided. `--level=anno` calculates the average time spent on each annotation.
Similarly, `--level=tile` and `level=pano` calculates the average time to
annotate an entire image and panorama, respectively.

#### Example 1: Total Time
The code block below shows an example of calculating the total time to annotate
a set of panoramas included in the inputted JSON file.
```commandline
python3 annotation_statistics.py\
--anno_list ../input/snb-5-annos.json \
--stat=time \
--level=all
```
```text
Total annotation time (seconds): 5.80
```

#### Example 2: Time per Panorama
The code block below shows an example that calculates the average time to fully
annotate a panorama from the inputted JSON file.
```commandline
python3 annotation_statistics.py \
--anno_list ../input/snb-5-annos.json \
--stat=time \
--level=pano
```
```text
Average annotation time per panorama (seconds): 2296.63
```
### Annotation Sizes
To find statistics related to the size of the annotations (as a percentage of
the tiles they are contained within) you need to specify `--stat=size`. 

#### Example 
The code block below shows an example that calculates the average & median
width, height, and size of annotations, as a percentage of the tile's width,
height, and size respectively. 
```commandline
python3 annotation_statistics.py \
--anno_list ../input/snb-5-annos.json \
--stat=size
```

```text
*** *** BBox Height (% of tile height) *** ***
    label  mean  median
Cormorant  5.47    5.04
     Nest  5.06    5.08

*** *** BBox Size (% of tile) *** ***
    label  mean  median
Cormorant  0.28    0.25
     Nest  0.58    0.53

*** *** BBox Width (% of tile width) *** ***
    label  mean  median
Cormorant  4.79    4.58
     Nest 10.21   10.64
```

