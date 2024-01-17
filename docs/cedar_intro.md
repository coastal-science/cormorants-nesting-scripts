# Cedar Cheatsheet

> [!WARNING]
> This document is not yet complete.

We assume you (1) have a Digital Alliance of Canada account, (2) you have been added to the appropriate projects (e.g. `ctb-ruthjoy`), and (3) you have Terminal or GitBash (or another  installed on your machine. 

## Accessing Cedar
Replace `user` with your Alliance username in the following command: 
`ssh user@cedar.computecanada.ca`

The very first time, you may be asked if you want to continue connecting. Type `yes` and hit enter. 

## Using the Command Line
#### Moving Around on Cedar
`pwd` - shows your **p**resent **w**orking **d**irectory.    
`ls` - **l**i**s**t contents of your current directory.   
`cd <filepath>` - **c**hange **d**irectory to the provided <filepath> e.g. `cd projects/ctb-ruthjoy/`.   
`[user@cedar5 workspace]$` - The start of the commandline gives you information on where you are located. Here, you are logged into `cedar5` as `user` and are inside the `workspace` directory.   

#### Inspecting Files
`cat <file>` - prints the file contents to screen.    
`tail` - prints the last 5 lines of the file to screen.    
`less` - opens a file, allows you to scroll around To exit, type `q`.   
`grep <pattern> <file>` - searches through a `file` to look for a `pattern`. Will print each line where the pattern is found.

#### Other Pointers
`Ctrl-C` - Interupt whatever command is currently running, and give you an empty command line
Up Arrow - Look through your previous commands
Tab - will auto-complete filepaths & filenames
`.` - refers to the current directory
`..` - refers to the parent directory
`~` - refers to your home directory
`*` - wildcard character


#### slurm commands
`sbatch <file.sh>` - submits a job to the slurm job scheduler. There are many options you can provide to SBATCH, which can be specified in the command line, or at the top of the batch script you are submitting to the scheduler.    
`sq` or `squeue -u <username>` - Shows what commands you currently have in the slurm queue. 
`sb` - provided you have run `source user_profile` this provides an alias for sbatch, that adds in your mail_user automatically.   
`scancel <job_number>` - Will stop the specified job from running.    

#### Using `nano`
`Ctrl-O` - write the file
`Ctrl-X` - exit the editor
