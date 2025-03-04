#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from utils.fid_score import calculate_fid_given_paths
from utils.mAPscore import  ap_per_class
from pytorch_msssim import ms_ssim

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)
                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                ms_ssim=[]

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    ms_ssim.append(ms_ssim(renders[idx],gts[idx],data_range=1.0).item())
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  MS-SSIM : {:>12.7f}".format(torch.tensor(ms_ssim)))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                fid_value=calculate_fid_given_paths(
                    [str(renders_dir),str(gt_dir)],
                    batch_size=10,
                    device="cuda",
                    dims=2048,
                    num_workers=4
                )
                print("   FID : {:>12.7f}",format(fid_value))

                # mAP 계산
                gt_boxes_path = method_dir / "gt_boxes.json"
                pred_boxes_path = method_dir / "pred_boxes.json"

                if gt_boxes_path.exists() and pred_boxes_path.exists():
                    with open(gt_boxes_path, "r") as f:
                        gt_boxes = json.load(f)
                    with open(pred_boxes_path, "r") as f:
                        pred_boxes = json.load(f)

                    conf_matrix = ConfusionMatrix(nc=len(gt_boxes))  # 클래스 수 지정
                    ious = []
                    for img_id in gt_boxes.keys():
                        if img_id in pred_boxes:
                            gt = torch.tensor(gt_boxes[img_id]).cuda()
                            pred = torch.tensor(pred_boxes[img_id]).cuda()
                            iou = box_iou(pred[:, :4], gt[:, :4])  # IoU 계산
                            ious.append(iou.mean().item())
                            conf_matrix.process_batch(pred, gt)  # Confusion matrix 업데이트

                                    # AP 및 mAP 계산
                    p, r, ap, f1, ap_class = ap_per_class(*conf_matrix.matrix)  # AP 계산
                    map50 = ap[:, 0].mean() if len(ap) else 0  # AP@0.5
                    map75 = ap[:, 1].mean() if len(ap) else 0  # AP@0.75
                    mean_map = ap.mean() if len(ap) else 0  # mAP@[0.5:0.95]

                    print(f"   mAP@0.5:  {map50:.7f}")
                    print(f"   mAP@0.75: {map75:.7f}")
                    print(f"   mAP@[.5:.95]: {mean_map:.7f}")


                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "MS-SSIM":torch.tensor(ms_ssim).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "mAP@0.5": map50,
                                                        "mAP@0.75": map75,
                                                        "mAP@[.5:.95]": mean_map},
                                                        "FID:",fid_value)
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "MS-SSIM":{name:ms_ssim for ms_ssim,name in zip(torch.tensor(ms_ssim).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
