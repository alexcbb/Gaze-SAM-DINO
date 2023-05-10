#!/bin/bash

#SBATCH --job-name=prepro_ycb
#SBATCH --output=logs/job.%j.out
#SBATCH --error=logs/job.%j.err 

#SBATCH --partition=haswell
#SBATCH --nodes=2
#SBATCH --mem-per-cpu=500M
#SBATCH --ntasks-per-node=5 #number of MPI tasks per node (=number of GPUs per node)
#SBATCH --cpus-per-task=1
#SBATCH -t 3:00:00
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL

# Mount sesali file
test -d ~/mnt/ycb-video || mkdir -p ~/mnt/ycb-video

sshfs sesali:/home/databases/ycb-video ~/mnt/ycb-video

python3 process_ycb.py --dataset_path ~/mnt/ycb-video/YCB_Video_Dataset --data_config config_cluster.yaml --prefix cluster_

# Unmount sesali file
fusermount -u ~/mnt/ycb-video
rmdir ~/mnt/ycb-video
