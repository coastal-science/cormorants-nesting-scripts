# Running a Trained Model on a New Panorama
 
 ## 1. Set up your Digital Research Alliance of Canada account
* Apply for a Digital Research Alliance of Canada account by visiting [this link](https://ccdb.alliancecan.ca/account_application). Unless you are a principal investigator, you will need to provide your sponsor's CCRI. Your sponsor can find their CCRI near the top of the page once they login to their CCDB account [here](https://ccdb.alliancecan.ca/). CCRIs follow the format `abc-123-01`. Once you've submitted your application your sponsor will receive an email with instructions on how to approve your application. 
* Once you have an Alliance account, contact a project owner or manager (e.g. Ruth or Jillian) and ask to be added to the `ctb-ruthjoy` project on cedar.
* If you encounter difficulties in setting up your account, send an email requesting support to <accounts@tech.alliancecan.ca>. 
  
## 2. Gather required files

### TIF file(s)
Using Globus, upload the original panorama image(s) (in TIF format) to the `CormorantNestingBC` collection. Follow the organizational scheme when choosing a name and location for the file(s).   

### Mask files 
If you will use masking to remove detections falling outside a particular region (e.g. IWMB Span 2), use the LOST annotation tool to generate a CSV file containing the mask annotations. Use Globus (or scp) to upload this CSV to Cedar. 
  
Rename this file to mask.csv and place it into the appropriate directory under `cormorant-nesting-scripts/object-detection-scripts/3_prediction_pipeline_postprocessing/post_process_detections/input/`. This directory should be the parent directory of the `task_path`s defined in the JSON file (described below). For example, from the `input` folder running:
  
```shell
$ head -2 ./2023/IWMB/snb5_cn_hg_v9/mask.csv
```

Will provide the following output: 

```CSV
img.idx,img.anno_task_id,img.timestamp,img.timestamp_lock,img.state,img.sim_class,img.frame_n,img.video_path,img.img_path,img.result_id,img.iteration,img.user_id,img.anno_time,img.lbl.idx,img.lbl.name,img.lbl.external_id,img.annotator,img.is_junk,anno.idx,anno.anno_task_id,anno.timestamp,anno.timestamp_lock,anno.state,anno.track_id,anno.dtype,anno.sim_class,anno.iteration,anno.user_id,anno.img_anno_id,anno.annotator,anno.confidence,anno.anno_time,anno.lbl.idx,anno.lbl.name,anno.lbl.external_id,anno.data
10003,108,42:52.4,43:26.2,4,1,,,data/media/2023_span2_masks/
20230710_IWMBSpan2_RockyPoint_gigapan_panorama_REDUCED_10.jpg,214,0,1,42.587,[],[],[],admin,,24593,108,42:52.4,42:09.6,4,,polygon,,0,1,10003,admin,,31.76,[55],"[""Bridge Mask""]","[""""]","[{""x"": 0.10922509225092251, ""y"": 0.962521664549575}, {""x"": 0.2199261992619926, ""y"": 0.9776869758689757}, {""x"": 0.2848708487084871, ""y"": 0.7524259319396716}, {""x"": 0.359409594095941, ""y"": 0.5921440352976668}, {""x"": 0.4199261992619926, ""y"": 0.4903433712142313}, {""x"": 0.566789667896679, ""y"": 0.3235635598434966}, {""x"": 0.7483394833948339, ""y"": 0.2152649810313312}, {""x"": 0.9985239852398524, ""y"": 0.10263445906667919}, {""x"": 0.9985239852398524, ""y"": 0.0051657381357303155}, {""x"": 0.7439114391143912, ""y"": 0.035489340203136634}, {""x"": 0.6029520295202953, ""y"": 0.11563028852413904}, {""x"": 0.46273062730627307, ""y"": 0.22392886733630443}, {""x"": 0.35424354243542433, ""y"": 0.34955521875841633}, {""x"": 0.2856088560885609, ""y"": 0.45568782599433844}, {""x"": 0.2177121771217712, ""y"": 0.6138037510600999}, {""x"": 0.17785977859778598, ""y"": 0.7112724719910487}, {""x"": 0.11512915129151291, ""y"": 0.9235376864628929}]"
```

### JSON file 
Create a JSON document which specifies image & task-path pairs. The file should follow the format shown below. This document should be uploaded to Cedar (via Globus or `scp`):
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

## 3. Run the model
* Using the commandline (via an application such as Terminal) login to cedar, using your own credentials (i.e. replace `user` in the command below) 
  ```shell
  ssh user@cedar.computecanada.ca
  ```

* Navigate to the cormorants workspace
  ```shell
  cd projects/ctb-ruthjoy/jilliana/Tensorflow/workspace
  ```

* If this is your first time running a model on Cedar, follow the instructions [below](#configure-your-user-profile) to configure your profile settings.
 
* Activate `user_profile.sh` by running `source user_profile.sh`.    

* Update `run_model_on_new_pano.sh` script using a text editor such as `nano` or `vim`. 
  ```shell
  nano run_model_on_new_pano.sh
  ```
  Read through the commands to ensure you understand what the script will be doing. Then, make any necessary changes.
  * `SBATCH` parameters - at the top of the file, you'll see a series of `#SBATCH` parameters. These define the resources requested by the job.
    * `--time=D-H:MM:SS` You will need approximately 15-20 minutes per panorama.
    * `--mail-user=email@address.ca` the email where notifications are sent.
  * `JSON_FILE` - change this to reference the JSON file corresponding to the panoramas you will be running through the model. 

* Submit the job to the batch scheduler (slurm)
  ```shell
  sb --time=1:0:0 run_model_on_new_pano.sh
  ```
  
* You will receive a response such as `Submitted batch job 123456789` which shows your job ID (in this case, `1234556789`).    

## 4. Monitoring your job
* Run `sq` to see your job's progress (is it pending, running, etc)
* If you've setup user_profile.sh & using the `sb` command, you should receive an email when your job starts, is cancelled, fails, or completes.
* Inspect the slurm logs using `cat`, `less`, `tail`, `head`, etc. Your slurm log will be found in a file name `slurm-<jobid>.out` (e.g. `slurm-123456789.out`). I find the following commands particularly useful: 
  * Determine which panorama is currently being processed (and which have already been processed):
    ```shell
    grep "LOG STATUS:" slurm-<jobid>.out
    ```
  * View the progress being made in processing a particular panorama. This will show a progress bar (including a time remaining estimate). Run the command again to see an updated progress bar. 
    ```shell
    tail -3 slurm-<jobid>.out; echo "\n"
    ```

## 5. Finding the Results
* You will receive an email when your job ends. This email will indicate
  whether the job completed successfully or encountered an error and failed along the way.
* If the job finished successfully, you can examine the output which will be saved in a sub-directory of `/projects/ctb-ruthjoy/jilliana/Tensorflow/workspace/cormorants-nesting-scripts/object_detection_scripts/3_prediction_pipeline_postprocessing/post_process_detections/output/`. 
* If there was an error that prevented the job from being successful, you can
  examine the bottom of the log file using `tail -3 slurm-<jobid>.out; echo "\n"` to see what went wrong.
  
----
### Configure Your User Profile
`nano user_profile.sh`
make sure there is an entry for yourself. If you have a sponsored account, your `SLURM_JOB_ACCOUNT` will be `def-sponsor`, where sponsor is your sponsor's username. If you do not have a sponsored account (i.e. are a PI) than your `SLURM_JOB_ACCOUNT` will be `def-user`, where `user` is your own username.
```bash
    user )
      export SLURM_JOB_ACCOUNT=def-ruthjoy
      export MAIL_USER=example@example.com
      ;;
```
