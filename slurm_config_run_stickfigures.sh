#!/usr/bin/env bash
#
#SBATCH --job-name=deep_imc_stickfigures
#SBATCH --output=./outputs/deep_imc_stickfigures_v1.txt
#SBATCH --ntasks=1
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

# debug info
hostname
which python3
nvidia-smi

env

# venv
source /home/wiss/xian/venvs/subspace_clustering_3_12/bin/activate
export BLAS=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3
export LAPACK=/usr/lib/x86_64-linux-gnu/lapack/liblapack.a
# pip install -U pip setuptools wheel
# train
python3 ./generic_self_expressive_multiview_clustering_senet_style.py --clusters=3,3 \
--pretrain-epochs=500 \
--joint-epochs=1000 \
--view-epochs=1000 \
--checkpoint=./outputs/stickfigures_senet.pt \
--visualization=./outputs/stickfigures_senet.html \
--dataset=stickfigures \
--augmentation-roles=upper,lower  \
--upper-lower-mask-strength=1.0  \
--tensorboard-log-dir=./outputs/runs_stickfigures/ \
--dataset-path=data/datasets/enrc_data/stickfigures >> ./outputs/deep_imc_stickfigures_out.txt

