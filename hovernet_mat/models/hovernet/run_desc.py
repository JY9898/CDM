import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from misc.utils import center_pad_to_shape, cropping_center
from .utils import crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss, nnbce_loss

from collections import OrderedDict

def get_expand_mask_sum(rgmask, h, w):
    num_zeros = torch.sum(rgmask == 0, dim=(1, 2))
    num_ones = torch.sum(rgmask == 1, dim=(1, 2))
    # print('num_zeros:',num_zeros)
    # print('num_ones:',num_ones)
    # 判断条件并生成新的张量
    # h, w = 4, 4  # 设定新张量的维度
    result_tensor = torch.zeros((rgmask.shape[0], h, w))

    for i in range(rgmask.shape[0]):
        if num_ones[i] > num_zeros[i]:
            result_tensor[i] = torch.ones((h, w))
        elif num_ones[i] < num_zeros[i]:
            result_tensor[i] = torch.zeros((h, w))
        else:  # 相等的情况
            result_tensor[i, :h // 2, :w // 2] = rgmask[i, 0, 0]  # 左上角
            result_tensor[i, :h // 2, w // 2:] = rgmask[i, 0, 1]  # 右上角
            result_tensor[i, h // 2:, :w // 2] = rgmask[i, 1, 0]  # 左下角
            result_tensor[i, h // 2:, w // 2:] = rgmask[i, 1, 1]  # 右下角
    return result_tensor

def get_expand_mask(rgmask, h, w):
    result_tensor = torch.zeros((rgmask.shape[0], h, w))

    for i in range(rgmask.shape[0]):
        result_tensor[i, :h // 2, :w // 2] = rgmask[i, 0, 0]  # 左上角
        result_tensor[i, :h // 2, w // 2:] = rgmask[i, 0, 1]  # 右上角
        result_tensor[i, h // 2:, :w // 2] = rgmask[i, 1, 0]  # 左下角
        result_tensor[i, h // 2:, w // 2:] = rgmask[i, 1, 1]  # 右下角
    return result_tensor
####
def train_step(batch_data, run_info):
    # TODO: synchronize the attach protocol.
    
    run_info, state_info = run_info
    # print("run_info:",run_info['net'].keys())
    # print("state_info:",state_info)
    loss_func_dict = {
        "bce": xentropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
        "nnbce": nnbce_loss,
    }

    # if run_info['net']['tag']==0:
    #     # print('train_step_oritrain_step_disc')
    #     result_dict = train_step_ori(batch_data, run_info, loss_func_dict)
    # elif run_info['net']["tag"]==1:

        # print('train_step_disc')
    result_dict = train_step_disc(batch_data, run_info, loss_func_dict)

    return result_dict

def train_step_disc(batch_data, run_info, loss_func_dict):
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {"EMA": {}}
    track_value = lambda name, value: result_dict["EMA"].update({name: value})
    ####
    model = run_info["net"]["desc"]
    optimizer = run_info["net"]["optimizer"]
    disc_model = run_info["net"]["disc"]
    disc_optimizer = run_info["net"]["disc_optimizer"]

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]
    mask = batch_data["mask"]
    expandrgmask = batch_data["rgmask"]   #e.g. [[0,1],[1,1]] 
    # expandrgmask = get_expand_mask(rgmask, mask.shape[1], mask.shape[2])

    # print('imgs.shape:',imgs.shape)  # torch.Size([B, 256, 256, 3])
    # print('true_np.shape:',true_np.shape)  # torch.Size([B, 164, 164])
    # print('true_hv.shape:',true_hv.shape)  # torch.Size([B, 164, 164, 2])
    # print('mask.shape:',mask.shape)  # torch.Size([B, 164, 164])
    # print('expandrgmask.shape:',expandrgmask.shape)  # torch.Size([B, 164, 164])
    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 5, 1)
    # plt.imshow(cv2.cvtColor((imgs[0,...]).numpy(), cv2.COLOR_BGR2RGB))
    # plt.title(f'IMG')
    # plt.axis('off')
    # plt.subplot(1, 5, 2)
    # plt.imshow(true_np[0,...])
    # plt.title(f'true_np')
    # plt.axis('off')
    # plt.subplot(1, 5, 3)
    # plt.imshow(true_hv[0,:,:,0])
    # plt.title(f'true_h')
    # plt.axis('off')
    # plt.subplot(1, 5, 4)
    # plt.imshow(mask[0,...])
    # plt.title(f'mask')
    # plt.axis('off')
    # plt.subplot(1, 5, 5)
    # plt.imshow(expandrgmask[0,...])
    # plt.title(f'expandrgmask')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(f'0_INPUT.png')
    # plt.close()
    # assert False

    imgs = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs = imgs.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = true_np.to("cuda").type(torch.int64)
    true_hv = true_hv.to("cuda").type(torch.float32)
    mask = mask.to("cuda").type(torch.int8)
    expandrgmask = expandrgmask.to("cuda").type(torch.float32)

    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
        "np": true_np_onehot,
        "hv": true_hv,
        "pd": expandrgmask,
    }

    if model.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
        true_tp_onehot = F.one_hot(true_tp, num_classes=model.nr_types)
        true_tp_onehot = true_tp_onehot.type(torch.float32)
        true_dict["tp"] = true_tp_onehot

    ####
    model.train()
    model.zero_grad()  # not rnn so not accumulate
    disc_model.train()
    disc_model.zero_grad()

    pred_dict = model(imgs)
    pred_dict['pd'] = disc_model(pred_dict['feat']).to(expandrgmask.dtype)  #([16, 164, 164])
    # showpd = pred_dict['pd'].clone().detach().cpu().numpy()
    # print('pred_dict[pd].shape:',pred_dict['pd'].shape)
    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 5, 5)
    # plt.imshow(showpd[0,...])
    # plt.title(f'{np.amin(showpd[0,...])},{np.amax(showpd[0,...])}')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(f'dis_pred.png')
    # plt.close()
    del pred_dict['feat']
    # for key in pred_dict:
    #     print(f"Key: {key}, Shape: {pred_dict[key].shape}")

    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )
    pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
    if model.nr_types is not None:
        pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)

    ####
    loss = 0
    loss_opts = run_info["net"]["extra_info"]["loss"]
    for branch_name in pred_dict.keys():
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [true_dict[branch_name], pred_dict[branch_name]]
            if loss_name == "msge":
                loss_args.append(true_np_onehot[..., 1])
            loss_args.append(mask)

            term_loss = loss_func(*loss_args)
            track_value("loss_%s_%s" % (branch_name, loss_name), term_loss.cpu().item())
            loss += loss_weight * term_loss

    track_value("overall_loss", loss.cpu().item())
    # * gradient update

    # torch.set_printoptions(precision=10)
    loss.backward()
    optimizer.step()
    disc_optimizer.step()
    ####

    # pick 2 random sample from the batch for visualization
    sample_indices = torch.randint(0, true_np.shape[0], (2,))

    imgs = (imgs[sample_indices]).byte()  # to uint8
    imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    pred_dict["np"] = pred_dict["np"][..., 1]  # return pos only
    pred_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in pred_dict.items()
    }

    true_dict["np"] = true_np
    true_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in true_dict.items()
    }

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict["raw"] = {  # protocol for contents exchange within `raw`
        "img": imgs,
        "np": (true_dict["np"], pred_dict["np"]),
        "hv": (true_dict["hv"], pred_dict["hv"]),
        "pd": (true_dict["pd"], pred_dict["pd"]),
    }
    return result_dict

def train_step_ori(batch_data, run_info, loss_func_dict):
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {"EMA": {}}
    track_value = lambda name, value: result_dict["EMA"].update({name: value})
    ####
    model = run_info["net"]["desc"]
    optimizer = run_info["net"]["optimizer"]

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]
    mask = batch_data["mask"]
    # print('imgs.shape:',imgs.shape)  # torch.Size([16, 256, 256, 3])
    # print('true_np.shape:',true_np.shape)  # torch.Size([16, 164, 164])
    # print('true_hv.shape:',true_hv.shape)  # torch.Size([16, 164, 164, 2])
    # print('mask.shape:',mask.shape)  # torch.Size([16, 164, 164])
    # assert False

    imgs = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs = imgs.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = true_np.to("cuda").type(torch.int64)
    true_hv = true_hv.to("cuda").type(torch.float32)
    mask = mask.to("cuda").type(torch.int8)

    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
        "np": true_np_onehot,
        "hv": true_hv,
    }

    if model.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
        true_tp_onehot = F.one_hot(true_tp, num_classes=model.nr_types)
        true_tp_onehot = true_tp_onehot.type(torch.float32)
        true_dict["tp"] = true_tp_onehot

    ####
    model.train()
    model.zero_grad()  # not rnn so not accumulate

    pred_dict = model(imgs)
    del pred_dict['feat']
    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )
    pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
    if model.nr_types is not None:
        pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)

    ####
    loss = 0
    loss_opts = run_info["net"]["extra_info"]["loss"]
    for branch_name in pred_dict.keys():
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [true_dict[branch_name], pred_dict[branch_name]]
            if loss_name == "msge":
                loss_args.append(true_np_onehot[..., 1])
            loss_args.append(mask)

            term_loss = loss_func(*loss_args)
            track_value("loss_%s_%s" % (branch_name, loss_name), term_loss.cpu().item())
            loss += loss_weight * term_loss

    track_value("overall_loss", loss.cpu().item())
    # * gradient update

    # torch.set_printoptions(precision=10)
    loss.backward()
    optimizer.step()
    ####

    # pick 2 random sample from the batch for visualization
    sample_indices = torch.randint(0, true_np.shape[0], (2,))

    imgs = (imgs[sample_indices]).byte()  # to uint8
    imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    pred_dict["np"] = pred_dict["np"][..., 1]  # return pos only
    pred_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in pred_dict.items()
    }

    true_dict["np"] = true_np
    true_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in true_dict.items()
    }

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict["raw"] = {  # protocol for contents exchange within `raw`
        "img": imgs,
        "np": (true_dict["np"], pred_dict["np"]),
        "hv": (true_dict["hv"], pred_dict["hv"]),
    }
    return result_dict

####
def valid_step(batch_data, run_info):
    run_info, state_info = run_info
    ####
    model = run_info["net"]["desc"]
    model.eval()  # infer mode

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs_gpu = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs_gpu = imgs_gpu.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = torch.squeeze(true_np).type(torch.int64)
    true_hv = torch.squeeze(true_hv).type(torch.float32)

    true_dict = {
        "np": true_np,
        "hv": true_hv,
    }

    if model.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).type(torch.int64)
        true_dict["tp"] = true_tp

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1]
        if model.nr_types is not None:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=False)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = {  # protocol for contents exchange within `raw`
        "raw": {
            "imgs": imgs.numpy(),
            "true_np": true_dict["np"].numpy(),
            "true_hv": true_dict["hv"].numpy(),
            "prob_np": pred_dict["np"].cpu().numpy(),
            "pred_hv": pred_dict["hv"].cpu().numpy(),
        }
    }
    if model.nr_types is not None:
        result_dict["raw"]["true_tp"] = true_dict["tp"].numpy()
        result_dict["raw"]["pred_tp"] = pred_dict["tp"].cpu().numpy()
    return result_dict


####
def infer_step(batch_data, model):

    ####
    patch_imgs = batch_data

    patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32)  # to NCHW
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

    ####
    model.eval()  # infer mode

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(patch_imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict.pop('feat', None)
        # for key, value in pred_dict.items():
        #     if hasattr(value, 'shape'):
        #         print(f"键: {key}，形状: {value.shape}")
        #     else:
        #         print(f"键: {key}，没有形状信息")
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1)

    # * Its up to user to define the protocol to process the raw output per step!
    return pred_output.cpu().numpy()


####
def viz_step_output(raw_data, nr_types=None):
    """
    `raw_data` will be implicitly provided in the similar format as the 
    return dict from train/valid step, but may have been accumulated across N running step
    """

    imgs = raw_data["img"]
    true_np, pred_np = raw_data["np"]
    true_hv, pred_hv = raw_data["hv"]
    if nr_types is not None:
        true_tp, pred_tp = raw_data["tp"]

    aligned_shape = [list(imgs.shape), list(true_np.shape), list(pred_np.shape)]
    aligned_shape = np.min(np.array(aligned_shape,dtype=object), axis=0)[1:3]

    cmap = plt.get_cmap("jet")

    def colorize(ch, vmin, vmax):
        """
        Will clamp value value outside the provided range to vmax and vmin
        """
        ch = np.squeeze(ch.astype("float32"))
        ch[ch > vmax] = vmax  # clamp value
        ch[ch < vmin] = vmin
        ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
        return ch_cmap

    viz_list = []
    for idx in range(imgs.shape[0]):
        # img = center_pad_to_shape(imgs[idx], aligned_shape)
        img = cropping_center(imgs[idx], aligned_shape)

        true_viz_list = [img]
        # cmap may randomly fails if of other types
        true_viz_list.append(colorize(true_np[idx], 0, 1))
        true_viz_list.append(colorize(true_hv[idx][..., 0], -1, 1))
        true_viz_list.append(colorize(true_hv[idx][..., 1], -1, 1))
        if nr_types is not None:  # TODO: a way to pass through external info
            true_viz_list.append(colorize(true_tp[idx], 0, nr_types))
        true_viz_list = np.concatenate(true_viz_list, axis=1)

        pred_viz_list = [img]
        # cmap may randomly fails if of other types
        pred_viz_list.append(colorize(pred_np[idx], 0, 1))
        pred_viz_list.append(colorize(pred_hv[idx][..., 0], -1, 1))
        pred_viz_list.append(colorize(pred_hv[idx][..., 1], -1, 1))
        if nr_types is not None:
            pred_viz_list.append(colorize(pred_tp[idx], 0, nr_types))
        pred_viz_list = np.concatenate(pred_viz_list, axis=1)

        viz_list.append(np.concatenate([true_viz_list, pred_viz_list], axis=0))
    viz_list = np.concatenate(viz_list, axis=0)
    return viz_list


####
from itertools import chain


def proc_valid_step_output(raw_data, nr_types=None):
    # TODO: add auto populate from main state track list
    track_dict = {"scalar": {}, "image": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    def _dice_info(true, pred, label):
        true = np.array(true == label, np.int32)
        pred = np.array(pred == label, np.int32)
        inter = (pred * true).sum()
        total = (pred + true).sum()
        return inter, total

    over_inter = 0
    over_total = 0
    over_correct = 0
    prob_np = raw_data["prob_np"]
    true_np = raw_data["true_np"]
    for idx in range(len(raw_data["true_np"])):
        patch_prob_np = prob_np[idx]
        patch_true_np = true_np[idx]
        patch_pred_np = np.array(patch_prob_np > 0.5, dtype=np.int32)
        inter, total = _dice_info(patch_true_np, patch_pred_np, 1)
        correct = (patch_pred_np == patch_true_np).sum()
        over_inter += inter
        over_total += total
        over_correct += correct
    nr_pixels = len(true_np) * np.size(true_np[0])
    acc_np = over_correct / nr_pixels
    dice_np = 2 * over_inter / (over_total + 1.0e-8)
    track_value("np_acc", acc_np, "scalar")
    track_value("np_dice", dice_np, "scalar")

    # * TP statistic
    if nr_types is not None:
        pred_tp = raw_data["pred_tp"]
        true_tp = raw_data["true_tp"]
        for type_id in range(0, nr_types):
            over_inter = 0
            over_total = 0
            for idx in range(len(raw_data["true_np"])):
                patch_pred_tp = pred_tp[idx]
                patch_true_tp = true_tp[idx]
                inter, total = _dice_info(patch_true_tp, patch_pred_tp, type_id)
                over_inter += inter
                over_total += total
            dice_tp = 2 * over_inter / (over_total + 1.0e-8)
            track_value("tp_dice_%d" % type_id, dice_tp, "scalar")

    # * HV regression statistic
    pred_hv = raw_data["pred_hv"]
    true_hv = raw_data["true_hv"]

    over_squared_error = 0
    for idx in range(len(raw_data["true_np"])):
        patch_pred_hv = pred_hv[idx]
        patch_true_hv = true_hv[idx]
        squared_error = patch_pred_hv - patch_true_hv
        squared_error = squared_error * squared_error
        over_squared_error += squared_error.sum()
    mse = over_squared_error / nr_pixels
    track_value("hv_mse", mse, "scalar")

    # *
    imgs = raw_data["imgs"]
    selected_idx = np.random.randint(0, len(imgs), size=(8,)).tolist()
    imgs = np.array([imgs[idx] for idx in selected_idx])
    true_np = np.array([true_np[idx] for idx in selected_idx])
    true_hv = np.array([true_hv[idx] for idx in selected_idx])
    prob_np = np.array([prob_np[idx] for idx in selected_idx])
    pred_hv = np.array([pred_hv[idx] for idx in selected_idx])
    viz_raw_data = {"img": imgs, "np": (true_np, prob_np), "hv": (true_hv, pred_hv)}

    if nr_types is not None:
        true_tp = np.array([true_tp[idx] for idx in selected_idx])
        pred_tp = np.array([pred_tp[idx] for idx in selected_idx])
        viz_raw_data["tp"] = (true_tp, pred_tp)
    viz_fig = viz_step_output(viz_raw_data, nr_types)
    track_dict["image"]["output"] = viz_fig

    return track_dict
