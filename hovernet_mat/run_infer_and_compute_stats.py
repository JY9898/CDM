"""run_infer.py

Usage:
  run_infer.py [options] [--help] <command> [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)

Options:
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name, 
                              and expected overlaid color. [default: '']

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used PanNuke and MoNuSAC, 
                              'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size per 1 GPU. [default: 32]
  --true_dir=<path>         True label dir

Two command mode are `tile` and `wsi` to enter corresponding inference mode
    tile  run the inference on tile
    wsi   run the inference on wsi

Use `run_infer.py <command> --help` to show their options and usage.
"""

tile_cli = """
Arguments for processing tiles.

usage:
    tile (--input_dir=<path>) (--output_dir=<path>) \
         [--draw_dot] [--save_qupath] [--save_raw_map] [--mem_usage=<n>]
    
options:
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --mem_usage=<n>        Declare how much memory (physical + swap) should be used for caching. 
                          By default it will load as many tiles as possible till reaching the 
                          declared limit. [default: 0.2]
   --draw_dot             To draw nuclei centroid on overlay. [default: False]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: False]
   --save_raw_map         To save raw prediction or not. [default: False]
"""

wsi_cli = """
Arguments for processing wsi

usage:
    wsi (--input_dir=<path>) (--output_dir=<path>) [--proc_mag=<n>]\
        [--cache_path=<path>] [--input_mask_dir=<path>] \
        [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] \
        [--save_thumb] [--save_mask]
    
options:
    --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --mask_dir=<path>       Path to directory containing tissue masks. 
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
    --save_thumb            To save thumb. [default: False]
    --save_mask             To save mask. [default: False]
"""
import time
import csv
import glob
import numpy as np
import torch
import logging
import os
import copy
from misc.utils import log_info
from docopt import docopt
import scipy.io as sio
from metrics.stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_dice_2,
    get_fast_pq,
    remap_label,
    pair_coordinates
)


def run_nuclei_type_stat(pred_dir, true_dir, type_uid_list=None, exhaustive=True):
    """GT must be exhaustively annotated for instance location (detection).

    Args:
        true_dir, pred_dir: Directory contains .mat annotation for each image. 
                            Each .mat must contain:
                    --`inst_centroid`: Nx2, contains N instance centroid
                                       of mass coordinates (X, Y)
                    --`inst_type`    : Nx1: type of each instance at each index
                    `inst_centroid` and `inst_type` must be aligned and each
                    index must be associated to the same instance
        type_uid_list : list of id for nuclei type which the score should be calculated.
                        Default to `None` means available nuclei type in GT.
        exhaustive : Flag to indicate whether GT is exhaustively labelled
                     for instance types
                     
    """
    file_list = glob.glob(pred_dir + "*.mat")
    file_list.sort()  # ensure same order [1]
    # print('--------------------------type--------------------------')
    # print(pred_dir)
    # print(true_dir)
    # print('file_list:',file_list)
    # print(len(file_list))

    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point
    for file_idx, filename in enumerate(file_list[:]):
        filename = os.path.basename(filename)
        # print(filename)
        basename = filename.split(".")[0]

        true_info = sio.loadmat(os.path.join(true_dir, basename + ".mat"))
        # dont squeeze, may be 1 instance exist
        true_centroid = (true_info["inst_centroid"]).astype("float32")
        true_inst_type = (true_info["inst_type"]).astype("int32")

        if true_centroid.shape[0] != 0:
            true_inst_type = true_inst_type[:, 0]
        else:  # no instance at all
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])

        # * for converting the GT type in CoNSeP, otherwise annotation
        true_inst_type[(true_inst_type == 3) | (true_inst_type == 4)] = 3
        true_inst_type[(true_inst_type == 5) | (true_inst_type == 6) | (true_inst_type == 7)] = 4

        pred_info = sio.loadmat(os.path.join(pred_dir, basename + ".mat"))
        # dont squeeze, may be 1 instance exist
        pred_centroid = (pred_info["inst_centroid"]).astype("float32")
        pred_inst_type = (pred_info["inst_type"]).astype("int32")

        if pred_centroid.shape[0] != 0:
            pred_inst_type = pred_inst_type[:, 0]
        else:  # no instance at all
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, 12
        )
        # print('paired:',paired)
        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            # print(paired)
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)

    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    ###
    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        f1_type = (2 * (tp_dt + tn_dt)) / (
            2 * (tp_dt + tn_dt)
            + w[0] * fp_dt
            + w[1] * fn_dt
            + w[2] * fp_d
            + w[3] * fn_d
        )
        return f1_type

    # overall
    # * quite meaningless for not exhaustive annotated dataset
    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0]
    fn_d = unpaired_true_type.shape[0]

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore

    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    w = [2, 2, 1, 1]

    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()

    results_list = [f1_d, acc_type]
    f1_types = []
    for type_uid in type_uid_list:
        f1_type = _f1_type(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        f1_types.append(f1_type)
        results_list.append(f1_type)
    results_list.append(np.mean(f1_types))
    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    # print('type:',np.array(results_list))
    return results_list


def run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=True, ext=".mat"):
    # print stats of each image
    # print(pred_dir)

    file_list = glob.glob("%s/*%s" % (pred_dir, ext))
    file_list.sort()  # ensure same order

    metrics = [[], [], [], [], []]
    for filename in file_list[:]:
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

        true = sio.loadmat(os.path.join(true_dir, basename + ".mat"))
        true = (true["inst_map"]).astype("int32")

        pred = sio.loadmat(os.path.join(pred_dir, basename + ".mat"))
        pred = (pred["inst_map"]).astype("int32")

        # to ensure that the instance numbering is contiguous
        pred = remap_label(pred, by_size=False)
        true = remap_label(true, by_size=False)

        # print(true)
        # print(pred)
        pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
        metrics[0].append(get_dice_1(true, pred))
        metrics[1].append(get_fast_aji_plus(true, pred))
        metrics[2].append(pq_info[0])  # dq
        metrics[3].append(pq_info[1])  # sq
        metrics[4].append(pq_info[2])  # pq

        if print_img_stats:
            # print(basename, end="\t")
            for scores in metrics:
                print("%f " % scores[-1], end="  ")
            print()
    ####
    metrics = np.array(metrics)
    print(metrics)
    metrics_avg = np.mean(metrics, axis=-1)
    # np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    # print('instance:',metrics_avg)
    metrics_avg = list(metrics_avg)
    return metrics_avg

#-------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    start_time = time.time()
        
    sub_cli_dict = {'tile' : tile_cli, 'wsi' : wsi_cli}
    start_ep = 35
    end_ep = 65
    stats_res = []
    for ep in range(start_ep,end_ep+1):
        print(f'====================={ep}=====================')
        args = docopt(__doc__, help=False, options_first=True, 
                        version='HoVer-Net Pytorch Inference v1.0')
        
        sub_cmd = args.pop('<command>')
        sub_cmd_args = args.pop('<args>')
        

        # ! TODO: where to save logging
        logging.basicConfig(
            level=logging.INFO,
            format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',datefmt='%Y-%m-%d|%H:%M:%S',
            handlers=[
                logging.FileHandler("debug.log"),
                logging.StreamHandler()
            ]
        )

        if args['--help'] and sub_cmd is not None:
            if sub_cmd in sub_cli_dict: 
                print(sub_cli_dict[sub_cmd])
            else:
                print(__doc__)
            exit()
        if args['--help'] or sub_cmd is None:
            print(__doc__)
            exit()

        sub_args = docopt(sub_cli_dict[sub_cmd], argv=sub_cmd_args, help=True)
        
        args.pop('--version')
        gpu_list = args.pop('--gpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

        nr_gpus = torch.cuda.device_count()
        log_info('Detect #GPUS: %d' % nr_gpus)

        args = {k.replace('--', '') : v for k, v in args.items()}
        sub_args = {k.replace('--', '') : v for k, v in sub_args.items()}

        
        args['model_path'] = f"{args['model_path']}/net_epoch={ep}.tar"
        print("model_path:",args['model_path'])
        if args['model_path'] == None:
            raise Exception('A model path must be supplied as an argument with --model_path.')

        nr_types = int(args['nr_types']) if int(args['nr_types']) > 0 else None
        method_args = {
            'method' : {
                'model_args' : {
                    'nr_types'   : nr_types,
                    'mode'       : args['model_mode'],
                },
                'model_path' : args['model_path'],
            },
            'type_info_path'  : None if args['type_info_path'] == '' \
                                else args['type_info_path'],
        }

        # ***
        run_args = {
            'batch_size' : int(args['batch_size']) * nr_gpus,

            'nr_inference_workers' : int(args['nr_inference_workers']),
            'nr_post_proc_workers' : int(args['nr_post_proc_workers']),
        }

        if args['model_mode'] == 'fast':
            run_args['patch_input_shape'] = 256
            run_args['patch_output_shape'] = 164
        else:
            run_args['patch_input_shape'] = 270
            run_args['patch_output_shape'] = 80

        if sub_cmd == 'tile':
            run_args.update({
                'input_dir'      : sub_args['input_dir'],
                'output_dir'     : f"{sub_args['output_dir']}/ep{ep}/",

                'mem_usage'   : float(sub_args['mem_usage']),
                'draw_dot'    : sub_args['draw_dot'],
                'save_qupath' : sub_args['save_qupath'],
                'save_raw_map': sub_args['save_raw_map'],
            })

        if sub_cmd == 'wsi':
            run_args.update({
                'input_dir'      : sub_args['input_dir'],
                'output_dir'     : f"{sub_args['output_dir']}/ep{ep}/",
                'input_mask_dir' : sub_args['input_mask_dir'],
                'cache_path'     : sub_args['cache_path'],

                'proc_mag'       : int(sub_args['proc_mag']),
                'ambiguous_size' : int(sub_args['ambiguous_size']),
                'chunk_shape'    : int(sub_args['chunk_shape']),
                'tile_shape'     : int(sub_args['tile_shape']),
                'save_thumb'     : sub_args['save_thumb'],
                'save_mask'      : sub_args['save_mask'],
            })
        # ***
        
        if sub_cmd == 'tile':
            from infer.tile import InferManager
            infer = InferManager(**method_args)
            infer.process_file_list(run_args)
        else:
            from infer.wsi import InferManager
            infer = InferManager(**method_args)
            infer.process_wsi_list(run_args)

        pred_dir = os.path.join(run_args['output_dir'],'mat/')
        true_dir = args['true_dir']
        print('pred_dir:',pred_dir)
        print('true_dir:',true_dir)

        inst_stat = run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=False)
        type_stat = run_nuclei_type_stat(pred_dir, true_dir)
        epres = [ep]
        epres.extend(inst_stat)
        epres.extend(type_stat)
        stats_res.append(epres)
        print(epres)
        # stats_res[ep] = {'inst':inst_stat, 'type':type_stat}

    csv_file = f"{sub_args['output_dir']}/results.csv"
    header = ['epoch','DICE','AJI','DQ','SQ','PQ','Fd','Acc','Fm','Fi','Fe','Fs','Fmean']
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(stats_res)

    print(f'Saved to {csv_file}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Run time: {elapsed_time} s")

