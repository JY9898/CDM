# Cascaded Diffusion Models for Augmenting Imbalanced Nuclei Classification
[Paper]()

## Abstract

We provide our PyTorch implementation of **Cascaded Diffusion Models for Augmenting Imbalanced Nuclei Classification**. 
In this paper, we propose a Cascaded Diffusion-based Sample Synthesis (CDS) framework for augmenting minority-class images.



## Prerequisites
- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN

## Dataset Preparation
Download the datasets following [HoverNet](https://github.com/vqdang/hover_net). Then use the following scripts to prepare the training datasets (need to change the directory path). This process will produce a npy folder and two csv(for the next training).
```bash
cd hovernet_mat/
python extract_patches.py 
```

## 1. Augmentation stage
### 1.1 Training
1) Train the rare-class generative model:
```bash
cd mask_syn/

CUDA_VISIBLE_DEVICES=0,1 OPENAI_LOGDIR="/path/diff_hvtp_rare1/" \
mpiexec -n 2 python mask_train.py \
  --data_dir '/path/rare_1.csv' \
  --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --image_size 256 \
  --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
  --resblock_updown True --use_fp16 True --use_scale_shift_norm False  --image_type rare \
  --max_iterations 200000 --batch_size 4 --lr 2e-5 --weight_decay 0.05 \
  --save_interval 50000 --viz_batch_size 4 --rare_class 1 
```

2) Train the multi-class generative model:
```bash
cd mask_syn/

CUDA_VISIBLE_DEVICES=0,1 OPENAI_LOGDIR="/path/diff_hvtps/" \
mpiexec -n 2 python mask_train.py \
  --data_dir '/path/multi.csv' \
  --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --image_size 256 \
  --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
  --resblock_updown True --use_fp16 True --use_scale_shift_norm False  --image_type multi \
  --max_iterations 200000 --batch_size 4 --lr 2e-5 --weight_decay 0.05 \
  --save_interval 50000 --viz_batch_size 4 --num_classes 5 
```

3) Train the mask-to-image generative model:
```bash
cd image_syn/

CUDA_VISIBLE_DEVICES=0,1 OPENAI_LOGDIR='/path/masktoimg/pretrain/' \
mpiexec -n 2 python image_train.py \
  --data_dir '/path/trainall_patch_npy/' \
  --lr 1e-4 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
  --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False \
  --use_scale_shift_norm True --use_checkpoint False --num_classes 5 --class_cond True --no_instance False \
  --max_iterations 200000 --save_interval 100000 --drop_rate 0.0 

CUDA_VISIBLE_DEVICES=0,1 OPENAI_LOGDIR='/path/masktoimg/finetune/' \
mpiexec -n 2 python image_train.py \
    --data_dir '/path/trainall_patch_npy/' \
    --lr 2e-5 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
    --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False \
    --use_scale_shift_norm True --use_checkpoint True --num_classes 5 --class_cond True --no_instance False \
    --max_iterations 400000 --save_interval 100000 --drop_rate 0.2 \
    --resume_checkpoint '/path/masktoimg/pretrain/model200000.pt'
```

### 1.2 Sampling 

1) Sample the rare-class mask images: Modify the config and run
```bash
CUDA_VISIBLE_DEVICES=0,1 OPENAI_LOGDIR="/path/test_sample_hvt1/" \
mpiexec -n 2 python rare_sample.py \
  --model_path '/path/diff_hvtp_rare1/ema_0.9999_200000.pt' \
  --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 \
  --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
  --resblock_updown True --use_fp16 True --use_scale_shift_norm False --channel_mult 1,1,2,3,4 \
  --batch_size 2 --predict_xstart False --num_samples 4 

python proc_rare.py
```
2) Sample the multi-class mask images:
```bash
CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python multi_sample.py --config config_multi/consep_multi.yaml

python proc_multi.py
```
3) Sample the conditional images:
```bash
CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python image_sample.py --config config_image/consep_halfp.yaml
```

## 2. Segmentation and classification stage
```bash
cd hovernet_mat/
```
### 2.1 Training
Modify and add the synthetic data path to the end of 'train_dir_list' in config.py. Then run:
```bash
python train.py --gpu=0
```

### 2.2 Infering
```bash
python run_infer_and_compute_stats.py --batch_size=128 --gpu=1 --model_mode="fast" \
  --true_dir="/data/dataset/datasetname/Test/Labels/" \
  --model_path="/data/dataset/datasetname/consep_hovernetmat_256x256_164x164/01/" \
  --nr_types=5 --type_info_path="./type_info_consep.json" \
  tile \
  --input_dir="/data/dataset/datasetname/Test/Images/" \
  --output_dir="/data/dataset/datasetname/consep_hovernetmat_256x256_164x164/inferres/"
```

## Apply pre-trained models and evaluate

### Pretrained Models
We provide pretrained models for each stage(Rare-Class Generative model, Multi-Class Generative model, Mask-to-Image Generative model, HoverNet-MAT model). You can download all pretrained models [here](https://pan.baidu.com/s/1S7U1JXXn_nCW9ZAy0zI-0g) (passcode: f0sw). The detailed parameters of the model are set in the respective config folders.


## Acknowledge
Our code is developed based on [guided-diffusion](https://github.com/openai/guided-diffusion), [Repaint](https://github.com/andreas128/RePaint) and [HoverNet](https://github.com/vqdang/hover_net). 
