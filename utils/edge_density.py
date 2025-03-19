import cv2
import numpy as np
import os
from glob import glob

def find_highest_edge_density(image_folder):
    """
    주어진 폴더에서 엣지 밀도가 가장 높은 이미지를 찾습니다.
    :param image_folder: 이미지가 저장된 폴더 경로
    :return: 엣지 밀도가 가장 높은 이미지 경로와 해당 밀도 값
    """
    image_paths = glob(os.path.join(image_folder, "*.jpg")) + glob(os.path.join(image_folder, "*.png"))
    
    if not image_paths:
        print("⚠️ No images found in the specified folder.")
        return None, 0

    max_edge_density = 0  # 최대 엣지 밀도
    max_edge_image = None  # 최대 엣지 밀도를 가진 이미지 경로

    for image_path in image_paths:
        img = cv2.imread(image_path, 0)  # 이미지를 그레이스케일로 로드
        if img is None:
            continue  # 이미지 로드 실패 시 건너뛰기

        blurred = cv2.GaussianBlur(img, (5, 5), 1.4)  # 가우시안 블러 적용 (노이즈 제거)
        edges = cv2.Canny(blurred, 100, 200)  # Canny 엣지 검출기 적용
        edge_density = np.sum(edges > 0) / edges.size  # 엣지 밀도 계산

        if edge_density > max_edge_density:
            max_edge_density = edge_density
            max_edge_image = image_path  # 파일 경로 저장

    return max_edge_image, max_edge_density

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Find the image with the highest edge density.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing images.")
    args = parser.parse_args()

    best_image, best_density = find_highest_edge_density(args.image_folder)
    if best_image:
        print(f"✅ 최댓값 이미지: {best_image}, 엣지 밀도: {best_density}")
