import cv2
import numpy as np
import os
from glob import glob
import argparse

# 명령줄에서 폴더 경로를 입력받음
parser = argparse.ArgumentParser(description="Find the image with the highest edge density.")
parser.add_argument("image_folder", type=str, help="Path to the folder containing images.")
args = parser.parse_args()

# 입력받은 폴더에서 이미지 파일 검색
image_paths = glob(os.path.join(args.image_folder, "*.jpg")) + glob(os.path.join(args.image_folder, "*.png"))

# 이미지가 없을 경우 경고 출력 후 종료
if not image_paths:
    print("No images found in the specified folder.")
    exit()

max_edge_density = 0  # 최대 엣지 밀도 저장
max_edge_image = None  # 최대 엣지 밀도를 가진 이미지 파일명

for image_path in image_paths:
    # 이미지 읽기 (그레이스케일)
    img = cv2.imread(image_path, 0)
    if img is None:
        continue  # 이미지 로드 실패 시 건너뛰기

    # Gaussian Blur 적용 (노이즈 제거)
    blurred = cv2.GaussianBlur(img, (5, 5), 1.4)

    # Canny Edge Detector 적용
    edges = cv2.Canny(blurred, 100, 200)

    # 엣지 밀도 계산
    edge_density = np.sum(edges > 0) / edges.size

    # 최대 엣지 밀도 업데이트
    if edge_density > max_edge_density:
        max_edge_density = edge_density
        max_edge_image = image_path  # 파일 경로 저장

# 최종 결과 출력
print("✅ 최대 엣지 밀도:", max_edge_density)
print("📌 가장 엣지 밀도가 높은 이미지:", max_edge_image)
