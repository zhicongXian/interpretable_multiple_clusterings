#!/usr/bin/env bash
#
#SBATCH --job-name=deep_imc_nr_objects_sensitivity_study
#SBATCH --output=./outputs/deep_imc_nr_objects_sensitivity_%A_%a.out
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

normalized_cut_weight=(0 0.2)
augmentation_consistency=(0.5 1)
cluster_diversity_weight=(0.0 0.1 0.2)
affinity_diversity_weight=(0.0 0.1 0.2)

combinations=()
for hp1 in "${normalized_cut_weight[@]}"; do
    for hp2 in "${augmentation_consistency[@]}"; do
        for hp3 in "${cluster_diversity_weight[@]}"; do
            for hp4 in "${affinity_diversity_weight[@]}"; do
                combinations+=("$hp1 $hp2 $hp3 $hp4")
            done
        done
    done
done

# Select this task's combination.
task_id=$SLURM_ARRAY_TASK_ID
if (( task_id >= ${#combinations[@]} )); then
    echo "Task index exceeds the number of combinations." >&2
    exit 1
fi

read -r hp1 hp2 hp3 hp4 <<< "${combinations[$task_id]}"

substring=normalized_cut_weight_${hp1}_augmentation_consistency_${hp2}_cluster_diversity_weight_${hp3}_affinity_diversity_weight_${hp4}
# train
python3 ./generic_self_expressive_multiview_clustering_senet_style.py --clusters=6,2,3 \
--pretrain-epochs=1000 \
--joint-epochs=2000 \
--view-epochs=3000 \
--batch-size=128 \
--checkpoint="./outputs/nr_objects_{$substring}.pt" \
--visualization="./outputs/nr_objects_{$substring}.html" \
--dataset=nr_objects \
--tensorboard-log-dir="./outputs/runs_nr_objects_{$substring}" \
--augmentation-roles=color,material,shape   \
--experiment-name="sensitivity_analysis__{$substring}" \
--loss-weights-config=configs/nr_objects_sweeping_on_different_effects.yaml \
--dataset-path=data/datasets/enrc_data/nr_objects

