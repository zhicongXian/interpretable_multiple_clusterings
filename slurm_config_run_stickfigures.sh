#!/usr/bin/env bash
#
#SBATCH --job-name=deep_imc_stickfigures
#SBATCH --output=./outputs/deep_imc_stickfigures.txt
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
python3 ./generic_self_expressive_multiview_clustering.py --clusters=3,3 \
--pretrain-epochs=5000 \
--joint-epochs=1000 \
--view-epochs=1000 \
--checkpoint=./outputs/stickfigures.pt \
--visualization=./outputs/stickfigures.html \
--dataset=stickfigures \
--tensorboard-log-dir=./outputs/runs/ \
--dataset-path=data/datasets/enrc_data/stickfigures >> ./outputs/deep_imc_stickfigures_out.txt

