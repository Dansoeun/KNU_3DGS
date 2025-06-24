import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    # 시간 측정을 위한 변수들
    total_time = 0
    rendering_times = []
    
    # 워밍업 (첫 번째 렌더링은 초기화 시간이 포함될 수 있음)
    print("워밍업 렌더링 수행 중...")
    if len(views) > 0:
        _ = render(views[0], gaussians, pipeline, background)
        torch.cuda.synchronize()  # GPU 작업 완료 대기
    
    print(f"[3DGS][{name}] 렌더링 시작...")
    for idx, view in enumerate(tqdm(views, desc="렌더링 진행 상황")):
        # CUDA 이벤트 방식으로 시간 측정 (보다 정확함)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # 렌더링 시간 측정 시작
        start_event.record()
        rendering = render(view, gaussians, pipeline, background)["render"]
        end_event.record()
        
        # GPU 작업 완료 대기
        torch.cuda.synchronize()
        
        # 밀리초 단위를 초 단위로 변환
        elapsed = start_event.elapsed_time(end_event) / 1000.0
        rendering_times.append(elapsed)
        total_time += elapsed
        
        # 개별 이미지 렌더링 시간 출력 (원하면 주석 해제)
        #print(f"[3DGS][{name}][{idx}] 이미지 추론 시간: {elapsed:.4f}초")

        # 이미지 저장
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    # 통계 계산
    avg_time = total_time / len(views) if views else 0
    if rendering_times:
        min_time = min(rendering_times)
        max_time = max(rendering_times)
    else:
        min_time = max_time = 0
    
    # 결과 출력
    print(f"\n[3DGS][{name}] 렌더링 시간 통계:")
    print(f"  - 총 이미지 수: {len(views)}")
    print(f"  - 평균 추론 시간: {avg_time:.4f}초")
    print(f"  - 최소 추론 시간: {min_time:.4f}초")
    print(f"  - 최대 추론 시간: {max_time:.4f}초")
    print(f"  - 총 렌더링 시간: {total_time:.4f}초")
    
    return avg_time, min_time, max_time, total_time

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        try:
            print("Scene 객체 초기화 중...")
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            print("Scene 객체 초기화 완료!")
        except Exception as e:
            print(f"⚠️ Scene 객체 초기화 실패: {e}")
            return

    print(f"📌 훈련 카메라 개수: {len(scene.getTrainCameras())}")
    print(f"📌 테스트 카메라 개수: {len(scene.getTestCameras())}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    results = {}

    if not skip_train:
        print("\n==== 훈련 세트 렌더링 시작 ====")
        stats = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        results["train"] = {
            "avg_time": stats[0],
            "min_time": stats[1],
            "max_time": stats[2],
            "total_time": stats[3],
            "image_count": len(scene.getTrainCameras())
        }

    if not skip_test:
        print("\n==== 테스트 세트 렌더링 시작 ====")
        stats = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        results["test"] = {
            "avg_time": stats[0],
            "min_time": stats[1],
            "max_time": stats[2],
            "total_time": stats[3],
            "image_count": len(scene.getTestCameras())
        }
    
    # 최종 결과 요약
    print("\n==== 최종 결과 요약 ====")
    for set_name, stats in results.items():
        print(f"[{set_name}] 평균 추론 시간: {stats['avg_time']:.4f}초 ({stats['image_count']} 이미지)")
    
    # 결과 파일로 저장 (선택사항)
    result_path = os.path.join(dataset.model_path, f"timing_results_{iteration}.txt")
    with open(result_path, 'w') as f:
        f.write(f"3DGS 렌더링 시간 측정 결과 (반복 {iteration})\n")
        f.write(f"날짜: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for set_name, stats in results.items():
            f.write(f"=== {set_name} 세트 ===\n")
            f.write(f"이미지 수: {stats['image_count']}\n")
            f.write(f"평균 추론 시간: {stats['avg_time']:.4f}초\n")
            f.write(f"최소 추론 시간: {stats['min_time']:.4f}초\n")
            f.write(f"최대 추론 시간: {stats['max_time']:.4f}초\n")
            f.write(f"총 렌더링 시간: {stats['total_time']:.4f}초\n\n")
    
    print(f"결과가 {result_path}에 저장되었습니다.")


if __name__ == "__main__":
    # 커맨드 라인 인자 파서 설정
    parser = ArgumentParser(description="테스트 스크립트 매개변수")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print(f"{args.model_path} 렌더링 중")

    # 시스템 상태 초기화 (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
