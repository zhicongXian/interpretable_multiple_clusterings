#!/usr/bin/env bash
#
#SBATCH --job-name=deep_imc_nr_objects
#SBATCH --output=./outputs/deep_imc_nr_objects.txt
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
python3 ./generic_self_expressive_multiview_clustering_senet_style.py --clusters=6,2,3 \
--pretrain-epochs=1000 \
--joint-epochs=1000 \
--view-epochs=1000 \
--checkpoint=./outputs/nr_objects_v1.pt \
--visualization=./outputs/nr_objects_v1.html \
--dataset=nr_objects \
--tensorboard-log-dir=./outputs/runs_nr_objects \
--augmentation-roles=color,shape,material  \
--experiment-name=with_augmentation
--dataset-path=data/datasets/enrc_data/nr_objects >> ./outputs/deep_imc_nr_objects_out.txt

