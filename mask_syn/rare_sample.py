"""
Generate a large batch of image samples from a model and save them as a large numpy array.
"""

import argparse
import os
from matplotlib import pyplot as plt

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def get_available_filename(base_filename, save_path, extension):
    filename = os.path.join(save_path, f"{base_filename}.{extension}")
    index = 1
    
    while os.path.exists(filename):
        filename = os.path.join(save_path, f"{base_filename}_{index}.{extension}")
        index += 1
    
    return filename

def main():
    args = create_argparser().parse_args()
    args.in_channels = 3

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    mask_path = os.path.join(logger.get_dir(), '0-hvt1npy')
    os.makedirs(mask_path, exist_ok=True)

    logger.log("sampling...")
    all_images_id = 0
    while all_images_id < args.num_samples:
        model_kwargs = {}
        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample,_ = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
        )

        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous().detach().cpu().numpy()

        for i in range(args.batch_size):
            hvt1 = sample[i,...]
            name = f'{all_images_id:05d}'
            svname = get_available_filename(name, mask_path, 'npy')
            np.save(svname, hvt1)
            
            all_images_id += 1
        
        logger.log(f"created {all_images_id} samples")
        print(f"created {all_images_id} samples")

    
    # dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=12000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
