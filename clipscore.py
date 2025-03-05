import clip
import torch
from PIL import Image
import os
import tqdm
import numpy as np
import warnings
from sklearn.preprocessing import normalize

def load_images_from_directory(image_dir):
    """지정된 디렉토리에서 모든 이미지 파일을 불러옵니다."""
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    return image_paths

'''
def extract_images(image_paths, model, preprocess, device):
    """이미지 특징 추출"""
    all_image_features = []
    with torch.no_grad():
        for path in tqdm.tqdm(image_paths, desc="Extracting Image Features"):
            try:
                # 실제 이미지 파일 로드 및 전처리
                image = preprocess(Image.open(path)).unsqueeze(0).to(device)  # 이미지 크기: [1, C, H, W]
                image_features = model.encode_image(image)  # 모델을 사용하여 이미지 특징 추출
                all_image_features.append(image_features)
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                continue
    
    # 특징 벡터 반환
    return all_image_features
'''

def extract_images(image_paths, model, preprocess, device):
    """이미지 특징 추출"""
    all_image_features = []
    with torch.no_grad():
        for path in tqdm.tqdm(image_paths, desc="Extracting Image Features"):
            try:
                # 실제 이미지 파일 로드 및 전처리
                image = preprocess(Image.open(path)).unsqueeze(0).to(device)  # 이미지 크기: [1, C, H, W]
                image_features = model.encode_image(image)  # 모델을 사용하여 이미지 특징 추출
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                all_image_features.append(image_features)
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                continue
    
    # 특징 벡터 반환
    return all_image_features

def extract_text(captions, model, device):
    """텍스트 특징 추출"""
    text_tokens = clip.tokenize(captions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features

'''
def get_clip_score(model, images, captions, device, w=2.5):
    """이미지-텍스트 CLIPScore 계산 (CLIP 모델을 사용)"""
    text_tokens = clip.tokenize(captions).to(device)  # Tokenize captions into tensor

    with torch.no_grad():
        # CLIP 모델에서 텍스트 특징을 추출
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 이미지와 텍스트의 유사도를 계산
        similarities = []
        for img_feature in images:
            # 이미지 특징 정규화
            img_feature /= img_feature.norm(dim=-1, keepdim=True)
            
            # 이미지와 각 텍스트 간의 코사인 유사도 계산
            similarity = torch.nn.functional.cosine_similarity(img_feature, text_features)
            similarities.append(similarity.mean().item())

    # 개별 이미지의 점수와 평균 점수 반환
    # NaN 값 처리
    if len(similarities) > 0:
        mean_score = np.mean([s for s in similarities if not np.isnan(s)])
    else:
        mean_score = 0

    return mean_score, similarities
'''

def get_clip_score(model, images, captions, device, w=2.5):
    """
    Improved CLIPScore calculation with Softmax conversion
    
    Args:
        model: CLIP model
        images: List of image features
        captions: List of text captions
        device: Computation device
        w: Scaling factor (default 2.5 as in original CLIPScore paper)
    
    Returns:
        Tuple of (mean_softmax_score, softmax_per_instance_scores)
    """
    # Handle empty captions
    if not captions or len(captions) == 0:
        captions = ["an image"] * len(images)
    
    # Ensure number of captions matches number of images
    if len(captions) < len(images):
        captions = captions * ((len(images) // len(captions)) + 1)
    captions = captions[:len(images)]
    
    with torch.no_grad():
        # Tokenize and encode text features
        text_tokens = clip.tokenize(captions).to(device)
        text_features = model.encode_text(text_tokens)
        
        # Normalize text features
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities and apply CLIPScore
        similarities = []
        for img_feature, caption in zip(images, captions):
            # Normalize image features
            img_feature /= img_feature.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity manually
            similarity = torch.nn.functional.cosine_similarity(img_feature, text_features)
            
            # Apply CLIPScore scaling
            clip_score = w * 100 * similarity.mean().item()
            similarities.append(clip_score)
        
        # Convert to numpy array for softmax
        clip_scores = np.array(similarities)
        
        # Apply softmax to convert scores to probabilities
        softmax_scores = torch.nn.functional.softmax(torch.tensor(clip_scores), dim=0).numpy()
        
        # Compute mean softmax score, handling potential NaN values
        mean_softmax_score = np.nanmean(softmax_scores) if len(softmax_scores) > 0 else 0
    
    return mean_softmax_score, softmax_scores.tolist()