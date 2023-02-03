# cormorants-nesting-scripts
This repository contains all the Python code relevant for the Cormorants BC Nesting project. 

The top-level directory contains two folders: 
1. [`gigapan_data_inventory`](/gigapan_data_inventory) &mdash; Contains a text file detailing what data we have for this project (last updated Feb 10, 2022). This includes reference to which data were used for training, validating, and testing the CNN models. 
2. [`object_detection_scripts`](/object_detection_scripts) &mdash; Contains Python code for the object detection pipeline. Within this directory, you'll find many more subdirectories, each corresponding to a particular task in the pipeline. Each subdirectory is composed of a `input`, `output`, and `src` folder. The structure of these directories is described in more detail within the `object_detection_scripts` [README.md](/object_detection_scripts/README.md). There you will also find instructions for running the Python code (on either your local machine or on a cluster system). 

## Using a Trained Model to Make Predictions
This assumes the TIF images have already been uploaded to Cedar. 

1. Log onto Cedar
    * `ssh <USER>@cedar.computecanada.ca`
    * When prompted, input your Compute Canada username & password
2. Navigate to the correct location
3. Update the `run_model_on_new_pano.sh` script
    * Open the file: `nano run_model_on_new_pano.sh`
    * Update `--mail-user`: This line might look like 
      `#SBATCH --mail-user=email@sfu.ca`. This tells the script where email
       notifications should be sent. Update this to be your email. 
    * Update `TASK_PATH`: This line might look like `TASK_PATH="YEAR/DATASET/XX_MONTH/SNB_YYYYMMDD/"`. This path tells the script where to place outputs (tiles, prediction results). Some steps in the script also use this path to determine where inputs are located (e.g. the prediction command uses the tiles as input). This path should be unique, otherwise you will overwrite previous results. Do not include spaces in this path. Whenever possible, follow the format given here. For example: `TASK_PATH="2020/TEST/08_August/SNB_20200803/"`. 
    * Update `IMAGE`: This line might look like 
      `IMAGE="/~/projects/ctb-ruthjoy/CormorantNestingBC/YYYY/IMG.tif`. This path 
      tells the script where it can find the original image being run through the 
      model. For example: `2020-GigaPan/SNB_03082020.tif`.   
    * Once you have finished these updates, close nano using `Ctrl+X`. You will
      be asked whether you want to save the file. Use `Y` to save.
4. Submit the job: 
    * From the `workspace` directory, run 
    `sbatch --account=def-ruthjoy run_model_on_new_pano.sh`. The job has been 
     successfully submitted when you receive a response such as 
     `Submitted batch job 44648250`. The number at the end of this confirmation
     is the Job ID. 
5. Wait for the job to begin: 
    * You will receive an email when the job starts. Depending on how busy the 
      cluster is you may have to wait for multiple hours. More often your job 
6. Monitor the job (optional)
  * To check how long your job has been running, type `squeue -u USER` (update `USER` with your own username). This will show you information on the job status. 
  * To check on the logs, run `tail -3 slurm-JOBID.out; echo "\n"` (update `JOBID` to the Job ID indicated when you submitted the job). Once the model is up and running you will see a progress bar that shows the estimated amount of time left. Run the command again to see an updated progress bar. 
7. Wait for the job to finish: 
  * You will receive an email when your job ends. This email will indicate
    whether the job completed successfully or encountered an error and failed along the way.
  * If the job finished successfully, you can examine the output (saved in 
    `~/projects/ctb-ruthjoy/jilliana/Tensorflow/workspace/cormorants-nesting-scripts/object_detection_scripts/predict/output/`). 
    If you have followup jobs to run, you can go back to step 2 and begin again.
  * If there was an error that prevented the job from being successful, you can
    examine the bottom of the log file using `tail -3 slurm-JOBID.out; echo "\n"` (update `JOBID` to the Job ID indicated when you submitted the job).
    
### Jillian to do
- [X] Add some sort of logging. Perhaps writing script file to the output. 
- [ ] Add in post-processing, with optional masks
- [X] Either delete tiled tif files or zip them up. 
