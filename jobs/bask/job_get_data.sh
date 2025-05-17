#!/bin/bash
#SBATCH --account=chenhp-data-gen
#SBATCH --qos=bham
#SBATCH --time=1:00:00
#SBATCH --nodes 1
#SBATCH --gres gpu:0
#SBATCH --gpus-per-task 0
#SBATCH --tasks-per-node 1
#SBATCH --constraint=a100_40
#SBATCH --mem=128G  # 请求内存

set -e
module purge
module load baskerville

# 运行 Python 命令
source /bask/projects/q/qingjiem-heart-tte/yifansun/conda/miniconda/etc/profile.d/conda.sh
conda init
conda activate echosyn
conda info --envs
cd /bask/projects/c/chenhp-data-gen/yifansun/project/EchoNet-Synthetic
bash scripts/extract_frames_from_videos.sh datasets/EchoNet-Dynamic/Videos data/vae_train_images/images/