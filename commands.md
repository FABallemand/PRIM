# Useful Commands for Télécom Paris GPU Cluster

## Connect to Cluster
```bash
ssh -Y fallemand-24@gpu-gw.enst.fr
```

## Job Submission

### Interactive Mode
Enter interactive mode for development and debugging only. This mode gives access to a console running on a computation node. Notebooks like Jupyter can be launched for easy interruption, correction and relaunch. If disconnected, the execution is interrupted, so using a screen console is recommended.
```bash
# Enter interactive mode
sinteractive --time (JJ-HH:MM:SS) --gpus (number of GPU, 20 CPU per GPU) --partition (A100 | V100 | P100 | A40 | mm)
sinteractive --time 00-10:00:00 --gpus 2 --partition V100

# Exit interactive mode
exit
```

### Wait Queue Mode
Enter wait queue mode for real computation (ex: training). Only bash scripts can be submited to the queue and must contain:
- A job configuration heading (name of job, requested resources...) in the form of a list of SLURM options preceded by the word #SBATCH
- The command lines to execute (loading env, launching the executable file using the srun command to accounbt for the sbatch configuration provided...)
```bash
# See available resources
sinfo

# Submit script to queue
sbatch <scriptname>

# Monitor jobs state
squeue -u $USER

# Print job submission parameters
scontrol show job <jobid>

# Cancel a job
scancel <jobid>
```

## Other Commands
```bash
# Copy folder/file from local to server
scp -r /path/to/local/dir user@remotehost:/path/to/remote/dir
scp -r /path/to/local/dir fallemand-24@gpu-gw.enst.fr:/path/to/remote/dir
```