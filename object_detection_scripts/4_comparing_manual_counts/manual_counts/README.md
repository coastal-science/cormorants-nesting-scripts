1. Annotations were completed by Rose, Ruth, and Rachel using LOST. Annotations 
 were done on 3000x3000 pixel tiles. 
2. Tiles were re-assembled and annotations were drawn on the full image. Rose 
reviewed these combined images to find erroneous and/or duplicate annotations
that should be removed from our final counts. Each annotation slated for removal
was noted in the `RemovalRecord.xlsx` file, with a reason for removal noted in
the third column (`duplicate over tiles`, `not a bird or nest`, or `outside 
masking area`). 
3. Finally, the `manual_counts.py` script was run to find the verified counts
that are used to evaluate model results. This script takes the annotations and
removes any of the annotations identified as errors or duplicates in step two. 
Then, annotations are grouped according to their parent labels (Cormorant or 
Nest) to give us our final count for each date. 
