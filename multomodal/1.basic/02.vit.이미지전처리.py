import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, requests
from io import BytesIO
from typing import List

def download_sample_images(save_dir:str)->List:
    '''샘플 이미지 다운로드'''
    sample_images = {
        'cat.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg',
        'dog.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg',
        'bird.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg/1200px-Eopsaltria_australis_-_Mogo_Campground.jpg',
    }
    downloaded_images = []
    os.makedirs(save_dir,exist_ok=True)
    for filename,url in sample_images.items():
        filepath = os.path.join(save_dir,filename)
        if os.path.exists(filepath):
            print(f'  이미존재 : {filepath}')
            downloaded_images.append(filepath)
            continue
        try:
            print(f' 다운로드 : {filename}')
            response = requests.get(url)
            response.raise_for_status()
            Image.open(BytesIO(response.content))
            image = Image.convert('RGB')
            image.save(filepath)
            print(f' 저장완료 : {filepath}')
            downloaded_images.append(filepath)
        except Exception as e:
            print(f' 다운로드 실패 : {e}')

    return downloaded_images

# 기본이미지 로딩
def basic_image_loading(image_path:str):
    '''기본 이미지 로딩 방법'''
    img = Image.open(image_path)
    print(f'[이미지 정보]')
    print(f'이미지 모드 : {img.mode}')
    print(f'이미지 크기 : {img.shape}')

    # numpy 배열로 변환
    img_array = np.array(img)
    print(f'[Numpy 배열 변환]')
    print(f'배열 사이즈 : {img_array.shape}')  # (H, W, C)
    print(f'데이터 타입 : {img_array.dtype}')
    print(f'값 범위 : {img_array.min()} ~ {img_array.max()}')
    
    return img

# viT 표준 전처리
def vi_standard_preprocessing(img):
    '''viT 표준 전처리 파이프라인'''
    IMAGE_SIZE = 244
    MEAN = [0.5, 0.5, 0.5]   # ImageNet 평균
    STD = [0.2, 0.2, 0.2]   # ImageNet 표준편차

    # 전처리 파이프라인
    preprocess =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    print(f'[전처리 결과]')
    print(f'원본이미지 크기 : {img.shape}')
    img_tensor = preprocess(img)
    print(f'전처리 후 크기 : {img_tensor.shape}')
    print(f'전처리 후 값 범위 : {img_tensor.min()} ~ {img_tensor.max()}')

    # 배치 차원 추가
    img_batch = img_tensor.unsquezze(0)
    print(f'배치처리 후 크기 : {img_batch.shape}')

    return img_tensor, preprocess

# 학습용 데이터 증강
def training_augmentation(img):
    """학습 시 사용하는 데이터 증강"""    
    
    IMAGE_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # 학습용 증강 파이프라인
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),  # 랜덤 크롭
        transforms.RandomHorizontalFlip(p=0.5),                      # 좌우 반전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 색상 변형
        transforms.RandomRotation(degrees=15),                       # 회전
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    # 평가용 파이프라인 (증강 없음)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    print(f"\n[학습용 증강 기법]")
    print(f"  1. RandomResizedCrop: 랜덤 위치/크기로 자르기")
    print(f"  2. RandomHorizontalFlip: 50% 확률로 좌우 반전")
    print(f"  3. ColorJitter: 밝기, 대비, 채도 변형")
    print(f"  4. RandomRotation: +-15도 회전")
    
    # 같은 이미지에 여러 번 증강 적용
    print(f"\n[동일 이미지에 증강 적용 예시]")
    augmented_images = []
    for i in range(4):
        aug_img = train_transform(img)
        augmented_images.append(aug_img)
        print(f"  증강 {i+1}: shape={aug_img.shape}, "
              f"min={aug_img.min():.3f}, max={aug_img.max():.3f}")
    
    return train_transform, val_transform, augmented_images

if __name__ == '__main__':
    sample_images = download_sample_images('./download_img')
    for img in sample_images:
        img = basic_image_loading(img)
        img_tensor, preprocess = vi_standard_preprocessing(img)
        train_transform, val_transform, augmented_images = training_augmentation(img)