# Running a Trained Model on a New Panorama

> [!WARNING]
> This document is not yet complete.
 
## 1. Gather required files
* TIF file - Using Globus, upload the full panorama image (in TIF format) to the `CormorantNestingBC` collection. Follow the organizational scheme when choosing a name and location for the file.
* Mask files - If you are planning to use masking to remove detections falling outside a particular region (e.g. IWMB Span 2), use the LOST annotation tool to generate a CSV file containing the mask annotations. Use Globus (or scp) to upload this CSV to Cedar. Rename this file to mask.csv and place it into each of the appropriate directories under `cormorant-nesting-scripts/object-detection-scripts/3_prediction_pipeline_postprocessing/post_process_detections/input/`. Note, a copy of this file should be made for each panorama being run through the model and placed into the appropriate sub-folder.  
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

## 2. Set up your Digital Research Alliance of Canada account
* Apply for a Digital Research Alliance of Canada account by visiting [this link](https://ccdb.alliancecan.ca/account_application). Unless you are a principal investigator, you will need to provide your sponsor's CCRI. Your sponsor can find their CCRI near the top of the page once they login to their CCDB account [here](https://ccdb.alliancecan.ca/). CCRIs follow the format `abc-123-01`. Once you've submitted your application your sponsor will receive an email with instructions on how to approve your application. 
* Once you have an Alliance account, contact a project owner or manager (e.g. Ruth or Jillian) and ask to be added to the `ctb-ruthjoy` project on cedar.
* If you encounter difficulties in setting up your account, send an email requesting support to <accounts@tech.alliancecan.ca>. 

## 3. Run the model
* Using the commandline (via an application such as Terminal) login to cedar, using your own credentials (i.e. replace `user` in the command below) 
  ```commandline
  ssh user@cedar.computecanada.ca
  ```
* Navigate to the cormorants workspace
  ```commandline
  cd projects/ctb-ruthjoy/jilliana/Tensorflow/workspace
  ```
* Edit the `run_model_on_new_pano.sh` script using a text editor such as `nano` or `vim`. 
  ```commandline
  nano run_model_on_new_pano.sh
  ```
  Read through the commands to ensure you understand what the script will be doing. Then, make any necessary changes.
  * `SBATCH` parameters - at the top of the file, you'll see a series of `#SBATCH` parameters. These define the resources requested by the job.
    * `--time=D-H:MM:SS` You will need approximately 15-20 minutes per panorama.
    * `--mail-user=email@address.ca` the email where notifications are sent.
  * `JSON_FILE` - change this to reference the JSON file corresponding to the panoramas you will be running through the model. 
