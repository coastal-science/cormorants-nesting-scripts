#!/bin/bash


fail(){
    printf '%s\n' "$1" >&2 ## Send message to stderr.
    exit "${2-1}" ## Return a code specified by $2, or 1 by default.
}


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
run_permission_fixer() {
  # Usage: 
  # To apply this function recursively:
  #     $ run_permission_fixer "-R" "your_target_group"
  # To apply this function non-recursively:
  #     $ run_permission_fixer "-1" "your_target_group"

  recurse=$1
  target_group=$2
  
  # Ensure we're operating on the current working directory
  echo "Operating in directory: $(pwd)"
  
  # Output file for logging purposes
  myprogramout=nohup_perm_fixer_$USER.out
  touch "$myprogramout"

  date --iso-8601=seconds | tee -a "$myprogramout"
  echo "Operating in directory: $(pwd)" | tee -a "$myprogramout"
  found=0
  total=0

  # Recursively or non-recursively process files
  if [[ "$recurse" == "-R" ]]; then
    echo "Changing group and permissions recursively" | tee -a "$myprogramout"

    # Find all files owned by $USER and $target_group
    # lfs find . -user "$USER" -group "$target_group" -type f \
    (lfs find . -user "$USER" -group "$target_group" -type d ; \
      lfs find . -user "$USER" -group "$target_group" -type f) \
      | while read -r file; do
      ((total++))
      # Check if the group has write permission using stat
      perm=$(stat -c "%a" "$file") # returs 3 digits with octal permissions
      # Extract the second digit (group permission) from the octal permissions
      group_perm=$(( (perm / 10) % 10 ))

      # Check if the group has write permission
      if [ $group_perm -lt 6 ]; then
        # If the numeric permission is less than 660, the group does not have write permission
        # 2750: Group is sticky 'drwxr-s---'
        # 660: Group has read and write permission. 'drwxrw----'
        # 644: Group has only read permission. 'drwxr--r--'

        echo "Updating permissions ($group_perm) for $file"
        echo "$perm, $group_perm: $file" | tee -a "$myprogramout"
        chmod g+rw "$file" | tee -a "$myprogramout"
        # chgrp "$target_group" "$file"
        ((found++))
      fi
    done
  
  else
    echo "Changing group and permissions non-recursively" | tee -a "$myprogramout"
    
    # Non-recursively find files in the current directory only
    # lfs find . -maxdepth 1 -user "$USER" -group "$target_group" -type f -type d \
    # | while read -r file; do
    (lfs find . -maxdepth 1  -user "$USER" -group "$target_group" -type d ; \
      lfs find . -maxdepth 1 -user "$USER" -group "$target_group" -type f) \
      | while read -r file; do
      ((total++))
      # Check if the group has write permission using stat
      perm=$(stat -c "%a" "$file") # returs 3 digits with octal permissions
      # Extract the second digit (group permission) from the octal permissions
      group_perm=$(( (perm / 10) % 10 ))

      # Check if the group has write permission
      if [ $group_perm -lt 6 ]; then
        # If the numeric permission is less than 660, the group does not have write permission
        # 2750: Group is sticky 'drwxr-s---'
        # 660: Group has read and write permission. 'drwxrw----'
        # 644: Group has only read permission. 'drwxr--r--'
        echo "Updating permissions for $file"
        echo "$file" | tee -a "$myprogramout"
        chmod g+rw "$file" | tee -a "$myprogramout"
        # chgrp "$target_group" "$file"
        ((found++))
      fi
    done
  fi

  echo "Permission fix completed. Updated $total files and folders. Output logged to $myprogramout" | tee -a "$myprogramout"
  echo "" | tee -a "$myprogramout"
  echo "" | tee -a "$myprogramout"
}

run_permission_fixer_interrupt() {
  target_group=$1

  # Ensure we're operating on the current working directory
  echo "Operating in directory: $(pwd)"
  
  # Output file for logging purposes
  myprogramout=nohup_perm_fixer_$USER.out
  touch "$myprogramout"
  
  date --iso-8601=seconds | tee -a "$myprogramout"
  echo "Operating in directory: $(pwd)" | tee -a "$myprogramout"

  # Log the permission change for nohup output
  echo "Changing permissions of $myprogramout" | tee -a "$myprogramout"
  chmod g+rw "$myprogramout"

  # Group and permissions update
  echo "Changing group and permissions of nohup_perm_fixer file" | tee -a "$myprogramout"

  lfs find "$myprogramout" -user "$USER" -group "$target_group" -type f | while read -r file; do
    
    perm=$(stat -c "%a" "$file") # returs 3 digits with octal permissions
    # Extract the second digit (group permission) from the octal permissions
    group_perm=$(( (perm / 10) % 10 ))

    # Check if the group has write permission
    if [ $group_perm -lt 6 ]; then
      # If the numeric permission is less than 660, the group does not have write permission
      # 2750: Group is sticky 'drwxr-s---'
      # 660: Group has read and write permission. 'drwxrw----'
      # 644: Group has only read permission. 'drwxr--r--'
      echo "Updating permissions for $file"
      echo "$file" | tee -a "$myprogramout"
      chmod g+rw "$file" | tee -a "$myprogramout"
      # chgrp "$target_group" "$file"
    fi
  done

  echo "Permission fix completed. Output logged to $myprogramout" | tee -a "$myprogramout"
}

count_file(){
  UNAME=${1:-$USER}
  target_group=${2:-$(stat -c "%G" .)} # group of pwd

  echo "Counting files owned by user=$UNAME and group=$target_group that does not have group 'w' permissions."
  # lfs find . -user "$UNAME" -type f | while read -r file; do
  # lfs find . -user "$UNAME" -group "$target_group" -type f -type d \ 
  found=0
  total=0

  (lfs find . -user "$USER" -group "$target_group" -type d ; \
    lfs find . -user "$USER" -group "$target_group" -type f) \
    | while read -r file; do
      ((total++))
      perm=$(stat -c "%a" "$file") # returs 3 digits with octal permissions
      # Extract the second digit (group permission) from the octal permissions
      group_perm=$(( (perm / 10) % 10 ))
      
      # Check if the group has write permission
      if [ $group_perm -lt 6 ]; then
        # If the numeric permission is less than 6, the group does not have write permission
        # 2750: Group is sticky 'drwxr-s---'
        # 660: Group has read and write permission. 'drwxrw----'
        # 654: Group has only read permission. 'drwxr-xr--'
        # 644: Group has only read permission. 'drwxr--r--'
        echo "$perm, $group_perm: $file" | tee -a nohup_perm_counter_$USER.out
        ((found++))
      fi
    done | tee -a nohup_perm_counter_$USER.out
    echo "Total files and folders found are $found / $total."
}

count_all(){
  UNAME=${1:-$USER}
  echo "Counting all files+folders with the command"
  echo "  \$ time lfs find . -type f -type d | wc -l"
  echo "Counting all files+folders owned by $UNAME with the comand"
  echo "  \$ time lfs find . -type f -type d -user $UNAME | wc -l"
}