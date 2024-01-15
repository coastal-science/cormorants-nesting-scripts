# Running a Trained Model on a New Panorama

## 1. Gather required files
* TIF file - Using Globus, upload the full panorama image (in TIF format) to the `CormorantNestingBC` collection. Follow the organizational scheme when choosing a name and location for the file.
* Mask files - If you are planning to use masking to remove detections falling outside a particular region (e.g. IWMB Span 2), use the LOST annotation tool to generate a CSV file containing the mask annotations. Use Globus (or scp) to
  upload this CSV to Cedar. Place the CSV into `cormorant-nesting-scripts/object-detection-scripts/4_manuscrpt_reporting` .
  ```commandline
  
  ```
* JSON file - Create a JSON document which specifies image & task-path pairs. The file should follow this format:
  ```json
  [
   {
      "task_path": "2023/IWMB/snb5_cn_hg_v9/20230405_IWMBSpan2",
      "image": "/project/6059501/CormorantNestingBC/NEW_ORGANIZATION/2023/IWMB/IWMBSpan2/20230405_IWMBSpan2_RockyPoint_gigapan_panorama.tif"
   },
   {
      "task_path": "2023/IWMB/snb5_cn_hg_v9/20230426_IWMBSpan2",
      "image": "/project/6059501/CormorantNestingBC/NEW_ORGANIZATION/2023/IWMB/IWMBSpan2/20230426_IWMBSpan2_RockyPoint_gigapan_panorama.tif"
   }
  ]
  ```

## 2. Signup for a Digital Research Alliance of Canada account
* 

## 3. Run the model

