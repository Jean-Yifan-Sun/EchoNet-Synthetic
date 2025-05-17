#!/bin/bash
#SBATCH --account=chenhp-data-gen
#SBATCH --qos=bham
#SBATCH --time=48:00:00
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --gpus-per-task 1
#SBATCH --tasks-per-node 1
#SBATCH --constraint=a100_40
#SBATCH --mem=156G  # 请求内存

set -e
module purge
module load baskerville

# 运行 Python 命令
source /bask/projects/q/qingjiem-heart-tte/yifansun/conda/miniconda/etc/profile.d/conda.sh
conda init
conda activate ldm
conda info --envs
cd /bask/projects/c/chenhp-data-gen/yifansun/project/EchoNet-Synthetic
cd external/stable-diffusion
export DATADIR=$(cd ../../data/vae_train_images && pwd)
python main.py \
    --base ../../echosyn/vae/usencoder_kl_16x16x4.yaml \
    -t True \
    --gpus 0, \
    --logdir experiments/vae \
    --resume experiments/vae/2025-05-13T19-00-43_usencoder_kl_16x16x4