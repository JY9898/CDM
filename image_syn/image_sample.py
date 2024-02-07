"""
Third stage: mask to image sampling.
"""

import os
import argparse
import shutil
import cv2
import torch as th
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from easydict import EasyDict as edict
from guided_diffusion import dist_util
from guided_diffusion.utils import load_config, get_device, load_state_dict, setup_seed
import torch.nn as nn

from guided_diffusion.script_util_repaint import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)  # noqa: E402
import pathlib
from guided_diffusion.datasets_image import load_data_smp

def monetoU8(sample):
    '''
    Input: [-1, 1]
    Output: [0, 255]
    '''
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample

def zonetoU8(sample):
    '''
    Input: [0, 1]
    Output: [0, 255]
    '''
    if sample is None:
        return sample

    sample = (sample * 255).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample

def get_available_filename(base_filename, save_path, extension):
    filename = os.path.join(save_path, f"{base_filename}.{extension}")
    index = 1
    
    while os.path.exists(filename):
        filename = os.path.join(save_path, f"{base_filename}_sdm{index}.{extension}")
        index += 1
    
    return pathlib.Path(filename).stem

def append_to_file(file_path, content):
    with open(file_path, 'a') as file:
        file.write(content)

def main(cfg):
    dist_util.setup_dist()
    device = dist_util.dev()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(cfg, model_and_diffusion_defaults().keys()), cfg=cfg
    )
    model.load_state_dict(dist_util.load_state_dict(cfg.model_path, map_location="cpu"))
    model.to(device)
    if cfg.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = cfg.show_progress 
    
    dl = load_data_smp(
        data_dir=cfg.img_dir,
        batch_size=cfg.batch_size,
        deterministic=False,
        random_rotate=True,
        random_flip=True,
    )

    sample_path = os.path.join(cfg.results_path, f'2-masktoimages_s{cfg.s}_{cfg.schedule_jump_params.t_T}_{cfg.trunc}')
    os.makedirs(sample_path, exist_ok=True)
    sampleimg_path = os.path.join(cfg.results_path, f'2-masktoimagesimgshow__s{cfg.s}_{cfg.schedule_jump_params.t_T}_{cfg.trunc}')
    os.makedirs(sampleimg_path, exist_ok=True)

    print("sampling...")
    allsmpnum = 0
    for (batch, cond) in iter(dl):
        batch = batch.type(th.float32)
        name = cond["name"]
        instmap = cond['label_ori']
        typemap = cond['label']
        # print('instmap.shape:',instmap.shape)
        print(name)
        model_kwargs = {}
        y = preprocess_input(cond, class_num=cfg.num_classes)
        model_kwargs["y"] = y.to(device)

        for k in cond.keys():
            if isinstance(cond[k], th.Tensor):
                cond[k] = cond[k].to(device)
        if isinstance(batch, th.Tensor):
            batch = batch.to(device)

        model_kwargs["gt"] = batch
        model_kwargs['gt_keep_mask'] = cond['mask']
        model_kwargs['s'] = cfg.s
        batch_size = model_kwargs["gt"].shape[0]

        sample_fn = sample_fn = (
            diffusion.p_sample_loop if not cfg.use_ddim else diffusion.ddim_sample_loop
        )

        allsmpnum += batch_size
        result = sample_fn(
            model,
            (batch_size, 3, cfg.image_size, cfg.image_size),
            clip_denoised=cfg.clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=show_progress,
            return_all=False,
            conf=cfg
        )
        srs = monetoU8(result)
        
        instmap = instmap.permute(0, 2, 3, 1).cpu().numpy()
        typemap = typemap.permute(0, 2, 3, 1).cpu().numpy()
        for k in range(0, batch_size):
            svim = srs[k].copy()
            newname = get_available_filename(name[k], sample_path, 'npy')

            svsample = np.concatenate([svim, instmap[k], typemap[k]], axis=-1)
            np.save(f'{sample_path}/{newname}.npy', svsample)
            svim = cv2.cvtColor(svim, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{sampleimg_path}/{newname}.png', svim)
        print(f'Sampled {allsmpnum}.')
        if allsmpnum >= cfg.num_samples:
            break

    # dist.barrier()
    print(f"sampling complete, created {allsmpnum} samples.")


def preprocess_input(data, class_num):
    # move to GPU and change data types
    data['label'] = data['label'].long()
    data['label_ori'] = data['label_ori'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    input_label = th.FloatTensor(bs, class_num, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    instance_edge_map = get_edges(data['label_ori'])
    input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

    return input_semantics


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()

if __name__ == "__main__":
    setup_seed(10)

    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)

    config.results_path = config.results_path + '/'
    os.makedirs(config.results_path, exist_ok=True)
    shutil.copy(args.config, config.results_path)
    print("Results saved in:",config.results_path)
    
    main(config)
# CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR='/mntcephfs/lab_data/jiyang/logs/DiffProj01/uncondiff_hvt1/sample_hvt1_10000/0-hvtps_fixtp1-large15-noaug/sdmhalfsmp/' \
# python sample_img_halfp.py \
#     --img_dir '/mntcephfs/lab_data/jiyang/logs/DiffProj01/uncondiff_hvt1/sample_hvt1_10000/0-hvtps_fixtp1-large15-noaug/3-fill_img/' \
#     --model_path '/mntcephfs/lab_data/jiyang/logs/Diffmix/sdm_finetune_ori/ema_0.9999_400000.pt' \
#     --results_path '/mntcephfs/lab_data/jiyang/logs/DiffProj01/uncondiff_hvt1/sample_hvt1_10000/0-hvtps_fixtp1-large15-noaug/' \
#     --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear \
#     --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True \
#     --num_classes 5 --is_train False --class_cond True --no_instance False --random_rotate True --random_flip True \
#     --batch_size 2 --s 2 --num_samples 2

# python sample_img_halfp.py --config config/samp_halfp.yaml