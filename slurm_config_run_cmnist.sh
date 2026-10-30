#!/usr/bin/env bash
#
#SBATCH --job-name=deep_imc_cmnist_single
#SBATCH --output=./outputs/deep_imc_cmnist_single.txt
#SBATCH --ntasks=1
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G

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
python3 ./generic_self_expressive_multiview_clustering_senet_style.py --clusters=10,10 \
--pretrain-epochs=1000 \
--joint-epochs=2000 \
--view-epochs=3000 \
--batch-size=64 \
--checkpoint=./outputs/cmnist.pt \
--visualization=./outputs/cmnist.html \
--dataset=cmnist \
--tensorboard-log-dir=./outputs/runs_cmnist \
--augmentation-roles=left,right  \
--experiment-name=first_run \
--loss-weights-config=configs/cmnist.yaml \
--dataset-path=data/datasets/enrc_data/cmnist >> ./outputs/deep_imc_cmnist_single_out.txt

