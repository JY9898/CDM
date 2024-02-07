#!/bin/bash
#SBATCH -J hvmix1
#SBATCH -N 1
#SBATCH -c 5
#SBATCH -p p-V100
#SBATCH --gpus-per-node=1
#SBATCH -A P00120220004

source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate torch310

python run_infer_and_compute_stats.py --batch_size=512 --gpu=0 \
    --true_dir='/mntcephfs/lab_data/jiyang/datasets/CoNSeP/Test/Labels/' \
    --model_mode="fast" \
    --model_path="/mntcephfs/lab_data/jiyang/logs/DiffProj01/hovernet_aug/consep_hovernet_aug_prob50_256x256_164x164/01/" \
    --nr_types=5 --type_info_path="/home/jiyang/CDMProj/hover_net_aug/type_info/type_info_consep.json" \
    tile \
    --input_dir="/mntcephfs/lab_data/jiyang/datasets/CoNSeP/Test/Images/" \
    --output_dir="/mntcephfs/lab_data/jiyang/logs/DiffProj01/hovernet_aug/consep_hovernet_aug_prob50_256x256_164x164/inferres/"