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
conda activate miccai24
conda info --envs
cd /bask/projects/c/chenhp-data-gen/yifansun/project/EchoNet-Synthetic
python echosyn/vae/eval_new/test_ssim.py \
    -real_data_path data/vae_train_images/LabeledImage \
    -fake_data_path samples/reconstructed/LabeledImage \
    -b 128 \