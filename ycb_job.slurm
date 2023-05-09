#!/bin/bash

#SBATCH --job-name=train_yolo
#SBATCH --output=logs/job.%j.out
#SBATCH --error=logs/job.%j.err 

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=1 #number of MPI tasks per node (=number of GPUs per node)
#SBATCH --cpus-per-task=1
#SBATCH -t 16:00:00
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL

# Mount sesali file
test -d ~/mnt/ycb-video || mkdir -p ~/mnt/ycb-video

sshfs sesali:/home/databases/ycb-video ~/mnt/ycb-video

# Python script to launch
python3 train_yolo.py --model_path yolov8n.pt --data_config config_data.yaml

# Unmount sesali file
fusermount -u ~/mnt/ycb-video
rmdir ~/mnt/ycb-video
