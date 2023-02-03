#!/bin/bash
# Helper functions for logging

section_break(){
  # $MODELDIR=$1
  OUTFILE=$1

  (echo && echo && \
  echo "==============================================" && \
  echo && echo ) >> "$OUTFILE"
}

get_unique_file(){
    # today=$( date +%Y%m%d )   # or: printf -v today '%(%Y%m%d)T' -1
    number=0

    # fname=$today.txt
    fname=$1
    file=${fname%.*} # get the filename without the extension 
    ext=${fname##*.} # get the extension
    
    # if fname already exists, then add a numeric suffix to the name.
    while [ -e "$fname" ]; do
        printf -v fname '%s-%02d.%s' "${file}" "$(( ++number ))" "${ext}"
    done

    printf 'Will use "%s" as filename\n' "${fname}"
    touch "$fname"
    # return "$fname"
}

progress(){
  # Write module list, environment variables and script to output
  OUTPUT_FOLDER=$1 # $MODELDIR or 'output/$TASK_PATH/''
  LOGSDIR=$2

  full_path=$(realpath $0) # $0 is the name of the current script as it was executed

  OUTPUT_FILE=${OUTPUT_FOLDER}/script-job.log
  get_unique_file $OUTPUT_FILE
  OUTPUT_FILE=$fname

  echo > $OUTPUT_FILE
  module list >> $OUTPUT_FILE

  section_break $OUTPUT_FILE

  # get environment variables that partially contain any of the following keywords
  printenv | grep -E 'NAME|ENVDIR|DEPS|LOGSDIR|SCRATCH|PROJECT|WORKSPACE|TF_MODEL_GARDEN|TF_OBJ_DET|TFHUB_CACHE_DIR|MODEL' \
    >> $OUTPUT_FILE

  section_break $OUTPUT_FILE
  echo >> $OUTPUT_FILE

  cat $full_path >> $OUTPUT_FILE
  cp $OUTPUT_FILE $LOGSDIR/ 

  echo Additional logs may be written to "$LOGSDIR" & echo
}

# progress "output/folder" "logs/folder"