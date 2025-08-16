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

# Reusable function to find files and directories
find_files() {
  local UNAME=${1:-$USER}
  local target_group=${2:-$(stat -c "%G" .)} # group of pwd
  local recurse=$3

  if [[ "$recurse" == "-R" ]]; then 
    # Recursively find files and directories
    #                     with -print0 for null-terminated output
    (lfs find . -user "$UNAME" -group "$target_group" -type d ; \
     lfs find . -user "$UNAME" -group "$target_group" -type f)
  else
    # Non-recursively find files and directories (maxdepth 1)
    (lfs find . -maxdepth 1 -user "$UNAME" -group "$target_group" -type d ; \
     lfs find . -maxdepth 1 -user "$UNAME" -group "$target_group" -type f)
  fi
}

# Reusable function to process files and update permissions or count
process_files() {
  local files=$1
  local change=$2
  found=0
  total=0

  # Output either "Changing permissions" or "Only counting" based on change flag
  [[ "$change" == "true" ]] && echo "Changing permissions." || echo "Not changing permission, only counting."

  # Process each file
  # while IFS= read -r -d '' file; do # null terminated strings
  while read -r file; do # new line teminated strings
    ((total++))
    perm=$(stat -c "%a" "$file")  # Get octal permissions
    group_perm=$(( (perm / 10) % 10 ))  # Extract the second digit (group permission)

    # Check if the group has write permission
    if [ $group_perm -lt 6 ]; then
        # 2750: Group is sticky 'drwxr-s---'
        # 660: Group has read and write permission. 'drwxrw----'
        # 644: Group has only read permission. 'drwxr--r--'

      echo "$perm, $group_perm: $file"
      if [[ "$change" == "true" ]]; then
        chmod g+rw "$file"
        # Uncomment the line below to change the group as well
        # chgrp "$target_group" "$file"
      fi
      ((found++))
    fi
  done <<< "$files"  # Here, we pass the files directly to the loop, 
  # pipe (|) creates a subshell and the counters do not propagate back their values.

  # Return the count of files processed
  total_dir=$(lfs find . | wc -l)
  echo "$found files counted/updated out of $total available: $total_dir all users."
}

run_permission_fixer() {
  # Usage
  # Args:
  #   1: group name. The default is the group owner of the pwd (.), `stat -c "%G"``
  #   2: '-R' for recusion. Any other value will only update the `log file` 'nohup_perm_fixer_$USER.out' 
  # To apply this function recursively:
  #     $ run_permission_fixer "your_target_group" "-R"
  # To apply this function non-recursively:
  #     $ run_permission_fixer "your_target_group"
  # 

  target_group=$1 # ${2:-$(stat -c "%G" .)} # group of pwd
  recurse=${2:--1}
  UNAME=$USER
  
  echo ""
  date --iso-8601=seconds
  echo "Operating in directory: $(pwd)"

  # Recursively or non-recursively find and process files
  echo "Changing group and permissions $([[ "$recurse" == "-R" ]] && echo "recursively" || echo "non-recursively")"

  echo "Finding files owned by user=$UNAME and group=$target_group that does not have group 'w' permissions, with recursion=$recurse."

  files=$(find_files $UNAME $target_group "$recurse")
  process_files "$files" "true" # Update permissions for found files

  echo "Permission fix completed."
  echo ""
  echo ""
}

run_permission_fixer_interrupt() {
  target_group=$1

  # Ensure we're operating on the current working directory
  echo "Operating in directory: $(pwd)"
  
  # Output file for logging purposes
  log_file=nohup_perm_fixer_$USER.out
  touch "$log_file"
  chmod g+w "$log_file"

  date --iso-8601=seconds
  echo "Operating in directory: $(pwd)"

  # Log the permission change for nohup output
  echo "Changing permissions of $log_file"
  chmod g+rw "$log_file"

  # Group and permissions update for the file
  echo "Changing group and permissions of nohup_perm_fixer file"

  # Process the file
  files=$(lfs find "$log_file" -user "$USER" -group "$target_group" -type f)
  process_files "$files" "true"

  echo "Permission fix completed. Output logged to $log_file"
}


# Function to count files without changing permissions (with reuse of find_files and process_files)
count_permission_files() {
  local UNAME=${1:-$USER}
  local target_group=${2:-$(stat -c "%G" .)} # group of pwd
  local recurse=${3:--R}  # Default to recursive if no argument is provided

  # Ensure we're operating on the current working directory
  echo ""
  date --iso-8601=seconds
  echo "Operating in directory: $(pwd)"
  
  echo "Finding files owned by user=$UNAME and group=$target_group that does not have group 'w' permissions, with recursion=$recurse."

  # Find files based on the recursion flag
  files=$(find_files $UNAME "$target_group" "$recurse")
  process_files "$files" "false"  # Only count, without changing permissions

  echo "Counting complete."
  echo ""
  echo ""
}

count_all(){
  UNAME=${1:-$USER}
  echo "Counting all files+folders with the command"
  echo "  \$ time lfs find . -type f -type d | wc -l"
  echo "Counting all files+folders owned by $UNAME with the comand"
  echo "  \$ time lfs find . -type f -type d -user $UNAME | wc -l"
}