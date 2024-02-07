"""
Second stage: multi-class image sampling.
"""

import os
import argparse
import shutil
import cv2
import pathlib
from matplotlib import pyplot as plt
import torch as th
import torch.nn.functional as F
import numpy as np


from easydict import EasyDict as edict
from guided_diffusion import dist_util
from guided_diffusion.utils import load_config, get_device, load_state_dict, setup_seed


from guided_diffusion.script_util_repaint import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)  # noqa: E402

from guided_diffusion.datasets_mask import load_data_smp


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

def tensortonp(sample):

    if sample is None:
        return sample
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample

def get_available_filename(base_filename, save_path, extension):
    filename = os.path.join(save_path, f"{base_filename}.{extension}")
    index = 1
    
    while os.path.exists(filename):
        filename = os.path.join(save_path, f"{base_filename}_tps{index}.{extension}")
        index += 1
    
    return pathlib.Path(filename).stem

def append_to_file(file_path, content):
    with open(file_path, 'a') as file:
        file.write(content)

def svhvtpsplt(mask,path):
    plt.figure(figsize=(14, 5))
    for i in range(6):
        plt.subplot(1, 6, i+1)
        plt.imshow(mask[:,:,i])
        plt.title(f'[{np.amin(mask[:,:,i]):.03f},{np.amax(mask[:,:,i]):.03f}]')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{path}')
    plt.close()
    return


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

    show_progress = cfg.show_progress  #
    print('rare_class:',cfg.rare_class)
    
 
    dl = load_data_smp(
        ann_dir=cfg.ann_dir,
        batch_size=cfg.batch_size,
        num_classes=cfg.num_classes,
        deterministic=cfg.deterministic,
        random_flip=cfg.random_flip,
        random_rotate=cfg.random_rotate,
        rare_class=cfg.rare_class,
        dila_size=cfg.dila_size,
    )


    # mask_path = os.path.join(cfg.results_path, '1-maskshow')
    # os.makedirs(mask_path, exist_ok=True)
    npy_path = os.path.join(cfg.results_path, '1-hvtpsnpy')
    os.makedirs(npy_path, exist_ok=True)
    
    print("sampling...")
    allsmpnum = 0
    for batch in iter(dl):
        model_kwargs = {}

        model_kwargs["gt"] = batch['hvtps'].to(device)  # [-1, 1], torch.Size([2, 6, 256, 256]), float32
        model_kwargs["gt_keep_mask"] = batch['mask'].to(device)  # [0, 1], torch.Size([2, 6, 256, 256]), float32
        hvtps = model_kwargs["gt"].clone().permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
        mask = model_kwargs["gt_keep_mask"].clone().permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
        # print(model_kwargs["gt"].shape,model_kwargs["gt_keep_mask"].shape)
        batch_size = model_kwargs["gt"].shape[0]

        name = batch["name"]
        # print(name)  # ['02673', '02710']
        # assert False
        sample_fn = (
            diffusion.p_sample_loop if not cfg.use_ddim else diffusion.ddim_sample_loop
        )

        # allsmpnum += batch_size
        samples = sample_fn(
            model,
            (batch_size, cfg.num_classes+1, cfg.image_size, cfg.image_size),
            clip_denoised=cfg.clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=show_progress,
            return_all=False,
            conf=cfg
        )

        samples = samples.permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()

        for k in range(0, batch_size):
            allsmpnum += 1
            svname = get_available_filename(name[k], npy_path, 'npy')
            # svhvtpsplt(samples[k,...],f'{mask_path}/{svname}.png')
            
            # image_dict = {
            #     # 'hvtps':hvtps[k,...],
            #     'masktp1':mask[k,...],
            #     'hvtps_pred':samples[k,...],
            # }
            np.save(f'{npy_path}/{svname}.npy', samples[k,...])
            
        print(f'sampled {allsmpnum}.')
        if allsmpnum >= cfg.num_samples:
            break

    print(f"sampling complete, created {allsmpnum} samples.")


if __name__ == "__main__":
    setup_seed(10)

    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)

    config.results_path = config.results_path + '/' + config.smpname + '/'
    os.makedirs(config.results_path, exist_ok=True)
    shutil.copy(args.config, config.results_path)
    print("Results saved in:",config.results_path)
    
    main(config)
