import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# 1. 패치분할
def patch_embedding():
    '''이미지를 패치로 분할하는 과정(patch embedding)'''
    # 설정
    image_isze = 224  # (224 x 224)
    patch_size = 16   # (16 x 16)
    channels = 3
    embedding_dim = 768

    # 패치수 계산
    num_patches = (image_isze // patch_size) **2
    print(f'    이미지크기 : {image_isze} x {image_isze}')
    print(f'    패치크기 : {patch_size} x {patch_size}')
    print(f'    채널수 : {channels}')
    print(f'    패치 수 : {image_isze // patch_size} x {image_isze // patch_size}')

    # 더미 이미지 생성
    dummy_image = torch.randn(1, channels, image_isze,image_isze)
    print(f'    더미 이미지 생성')
    print(f'    입력 이미지 shape : {dummy_image.shape}')  # [1, 3, 224, 224]

    # 패치분할(Conv2d 사용)
    # Conv2d stride = patch_size 겹치지 않는 패치 추출
    patch_embed = nn.Conv2d(in_channels=channels, out_channels=embedding_dim,kernel_size=patch_size,stride=patch_size)

    # 패치 임베딩 적용
    patches =  patch_embed(dummy_image)
    print(f'\n패치임베딩 후')
    print(f'    Conv2d 출력 sahpe : {patches.shape}')  # [1, 768, 14 , 14]

    # Flatten : (B,D,H,W) -> (B, N, D)  (1,196,768)
    patches_flat = patches.flatten(2).transpose(1,2)
    print(f'    Flatten 후 sahpe : {patches_flat.shape}')  # [1, 196, 768]

    # 각 패치는 768차원 벡터
    print(f'   \n패치수 : {patches_flat.shape[1]}')
    print(f'   각 패치의 임베딩 차원 수 : {patches_flat.shape[2]}')
    return patches_flat


# 위치임베딩의 역활
def positional_embedding():
    '''위치 임베딩'''
    num_patches = 196
    embedding_dim = 768

    # 위치 임베딩 생성
    # 이 텐서는 학습대상 Optimizer의해 업데이트
    position_embedding =  nn.Parameter( torch.randn(1, num_patches+1,embedding_dim))   # +1은 CLS 토큰
    print(f'    위치 임베딩 shape : {position_embedding.shape}')
    print(f'    총 위치수 : {num_patches+1}  (패치 196 + cls토큰 1)')
    # 배치차원 제거  :각 위치를 하나의 벡터로 다루기위해 배치크기가 1인 형태는 분석시 불 필요
    pos_emb = position_embedding.squeeze(0)



if __name__=='__main__':
    patch_embedding()


