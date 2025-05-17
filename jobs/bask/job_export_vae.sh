#!/bin/bash
#SBATCH --account=chenhp-data-gen
#SBATCH --qos=bham
#SBATCH --time=4:00:00
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --gpus-per-task 1
#SBATCH --tasks-per-node 1
#SBATCH --constraint=a100_40
#SBATCH --mem=56G  # 请求内存

set -e
module purge
module load baskerville

# 运行 Python 命令
source /bask/projects/q/qingjiem-heart-tte/yifansun/conda/miniconda/etc/profile.d/conda.sh
conda init
conda activate ldm
conda info --envs
cd /bask/projects/c/chenhp-data-gen/yifansun/project/EchoNet-Synthetic
python scripts/convert_vae_pt_to_diffusers.py \
    --vae_pt_path /bask/projects/c/chenhp-data-gen/yifansun/project/EchoNet-Synthetic/external/stable-diffusion/experiments/vae/2025-05-13T19-00-43_usencoder_kl_16x16x4/checkpoints/last.ckpt \
    --dump_path /bask/projects/c/chenhp-data-gen/yifansun/project/EchoNet-Synthetic/models/vae/ \
    