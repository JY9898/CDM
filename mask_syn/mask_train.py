"""
Train a diffusion model for nuclei structure.

The nuclei structure consist of horizontal distance map, vertical distance map and minority mask.
"""

import argparse, os, sys
from guided_diffusion import dist_util, logger
from guided_diffusion.datasets_mask import load_data_rm, load_data_mm
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

def main():
    args = create_argparser().parse_args()
    if args.image_type=="rare":
        args.in_channels = 3
        logger.log("creating rareclass dataloader...")
        data = load_data_rm(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            rare_class=args.rare_class,
            random_rotate=True,
            random_flip=True,
        )
        
    elif args.image_type=="multi":
        args.in_channels = args.num_classes + 1
        logger.log("creating multiclass dataloader...")
        data = load_data_mm(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            random_rotate=True,
            random_flip=True,
        )
    else:
        raise ValueError(f"unspecified datatype {args.image_type}")
    
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        image_size=args.image_size,
        batch_size=args.batch_size,
        viz_batch_size=args.viz_batch_size,
        microbatch=args.microbatch,
        inpch=args.in_channels,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        max_iterations=args.max_iterations,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        viz_batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        max_iterations=100000,
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        num_classes=5,
        rare_class=1,
        in_channels=3,
        image_type="rare",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
