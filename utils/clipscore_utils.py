'''
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
'''
import clip
import torch
from PIL import Image
import os
import tqdm
import numpy as np
import warnings
from sklearn.preprocessing import normalize

def load_images_from_directory(image_dir):
    """Load all image files from a specified directory."""
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    return image_paths

def extract_images(image_paths, model, preprocess, device):
    """Extract image features"""
    all_image_features = []
    with torch.no_grad():
        for path in tqdm.tqdm(image_paths, desc="Extracting Image Features"):
            try:
                # Load and preprocess image
                image = preprocess(Image.open(path)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                all_image_features.append(image_features)
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                continue
    
    return all_image_features

def get_clip_score(model, images, captions, device, w=2.5):
    """
    CLIPScore calculation with explicit softmax conversion
    
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
        captions = [] * len(images)
    
    # Ensure number of captions matches number of images
    if len(captions) < len(images):
        captions = captions * ((len(images) // len(captions)) + 1)
    captions = captions[:len(images)]
    
    with torch.no_grad():
        # Tokenize and encode text features
        text_tokens = clip.tokenize(captions).to(device)
        text_features = model.encode_text(text_tokens)
        
        # Normalize text features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute raw similarities
        raw_similarities = []
        for img_feature, text_feature in zip(images, text_features):
            # Compute cosine similarity 
            similarity = torch.nn.functional.cosine_similarity(img_feature, text_feature)
            
            # Apply CLIPScore scaling
            clip_score = w * 100 * similarity.item()
            raw_similarities.append(clip_score)
        
        # Convert raw similarities to probabilities using softmax
        softmax_scores = torch.nn.functional.softmax(torch.tensor(raw_similarities), dim=0).numpy()
        
        # Compute mean softmax score
        mean_softmax_score = np.mean(softmax_scores)
    
    return mean_softmax_score, softmax_scores.tolist()