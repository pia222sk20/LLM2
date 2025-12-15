# huggingface transformer vit
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os, json

def use_huggingface_vit():
    '''huggingface transformers vit'''
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        # 모델과 프로세스 로드
        model_name = 'google/vit-base-patch16-224'
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        model.eval()
        print(f'\n[모델 정보]')
        print(f'파라메터수 : { sum(p.numel()  for p in model.parameters())}')
        print(f'클래스 수 : { model.config.num_channels}')
        print(f'이미지 크기 : { model.config.image_size}')
        print(f'패치 크기 : { model.config.patch_size}')
        print(f'히든 크기 : { model.config.hidden_size}')
        print(f'레이어 수 : { model.config.num_hidden_layers}')
        print(f'어텐션 해드 수 : { model.config.num_attention_heads}')
        return model, processor
    except Exception as e:
        print(f' hugging face vit 로드 실패 : {e}')
        return None, None
# timm 라이브러리를 사용한 vit
def use_timm_vit():
    '''timm 라이브러리 vit'''    
    import timm
    # 사용가능한 vit 모델 목록
    vit_models = timm.list_models('vit*', pretrained=True)
    for model_name in vit_models[:10]:
        print(f'    - {model_name}')
    print(f'총   {len(vit_models)}개 모델')

if __name__=='__main__':
    use_timm_vit()