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
import clip
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm.auto import tqdm  # Use tqdm.auto
from utils.image_utils import psnr
from argparse import ArgumentParser
from utils.fid_score import calculate_fid_given_paths
from utils.clipscore_utils import (
    load_images_from_directory, 
    extract_images, 
    get_clip_score
)
from utils.edge_density import find_highest_edge_density
from pytorch_msssim import ms_ssim as ms_ssim_func

# torch.Tensor를 JSON 직렬화 가능한 리스트로 변환
def tensor_to_list(tensor):
    return tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor

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
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            # CLIP 모델 로드
            device = torch.device("cuda:0")
            model, preprocess = clip.load("ViT-B/32", device=device)

            for method in os.listdir(test_dir):
                print("Method:", method)
                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                # 엣지 밀도가 가장 높은 이미지 찾기
                best_image, best_density = find_highest_edge_density(renders_dir)
                if best_image:
                    print(f"최댓값 이미지: {best_image}\n, 엣지 밀도: {best_density}")

                # CLIPScore 계산
                image_paths = load_images_from_directory(renders_dir)
                
                # 사용자로부터 캡션 입력 받기 
                print("\n이미지들에 대한 설명(캡션)을 입력해주세요.")
                print("1. 각 이미지마다 다른 캡션을 적용하려면 쉼표(,)로 구분된 캡션들을 입력하세요.")
                
                user_captions = input("캡션 입력: ").strip()
                
                # 캡션 처리
                if user_captions:
                    captions = [caption.strip() for caption in user_captions.split(',')]
                    if len(captions) < len(image_paths):
                        captions = (captions * ((len(image_paths) // len(captions)) + 1))[:len(image_paths)]
                else:
                    captions = [] * len(image_paths)

                image_features = extract_images(image_paths, model, preprocess, device)
                mean_score, per_instance_scores = get_clip_score(model, image_features, captions, device)

                # CLIPScore 출력
                print(f" 평균 CLIPScore: {mean_score:.4f}")

                ssims = []
                psnrs = []
                lpipss = []
                ms_ssims = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    ms_ssims.append(ms_ssim_func(renders[idx], gts[idx], data_range=1.0).item())
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean().item()))
                print("  MS-SSIM : {:>12.7f}".format(torch.tensor(ms_ssims).mean().item()))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean().item()))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean().item()))

                fid_value = calculate_fid_given_paths(
                    [str(renders_dir), str(gt_dir)],
                    batch_size=10,
                    device="cuda",
                    dims=2048,
                    num_workers=4
                )
                print("   FID : {:>12.7f}".format(fid_value))

                full_dict[scene_dir][method].update({
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                    "FID": fid_value,
                    "CLIPScore": mean_score
                })

                per_view_dict[scene_dir][method].update({
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "MS-SSIM": {name: ms_ssim for ms_ssim, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                    "CLIPScore": {name: score for name, score in zip(image_names, per_instance_scores)}
                })

            # JSON 파일 저장 시 torch.Tensor 처리
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump({k: tensor_to_list(v) for k, v in full_dict[scene_dir].items()}, fp, indent=True)

            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump({k: tensor_to_list(v) for k, v in per_view_dict[scene_dir].items()}, fp, indent=True)

        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)