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
    for model_name in vit_models:
        if 'vit_base_patch16_224' in model_name:
            print(f'    - {model_name}')
    print(f'총   {len(vit_models)}개 모델')

    model = timm.create_model('vit_base_patch16_224',pretrained=True)
    model.eval()
    print(f"실제 다운로드 모델명 : vit_base_patch16_224.{model.default_cfg['tag']}")

    # timm의 데이터 설정 가져오기
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return model, transform

def classify_image_hf(model, processor, image):
    """Hugging Face 모델로 이미지 분류"""   
    
    if model is None:
        print("  모델이 로드되지 않았습니다.")
        return None
    
    # 이미지 전처리
    inputs = processor(images=image, return_tensors="pt")
    print(f"\n[전처리된 입력]")
    print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
    
    # 추론
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    print(f"\n[모델 출력]")
    print(f"  logits shape: {logits.shape}")
    
    # Top-5 예측
    probs = F.softmax(logits, dim=-1)
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    print(f"\n[Top-5 예측 결과]")
    for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
        label = model.config.id2label[idx.item()]
        print(f"  {i+1}. {label}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
    
    return top5_probs[0], top5_indices[0]


def classify_image_timm(model, transform, image):
    """timm 모델로 이미지 분류"""    
    
    if model is None:
        print("  모델이 로드되지 않았습니다.")
        return None
    
    # 이미지 전처리
    img_tensor = transform(image).unsqueeze(0)
    print(f"\n[전처리된 입력]")
    print(f"  tensor shape: {img_tensor.shape}")
    
    # 추론
    with torch.no_grad():
        outputs = model(img_tensor)
    
    print(f"\n[모델 출력]")
    print(f"  outputs shape: {outputs.shape}")
    
    # Top-5 예측
    probs = F.softmax(outputs, dim=-1)
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    # ImageNet 클래스 이름 로드
    try:
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url, timeout=10)
        categories = [s.strip() for s in response.text.splitlines()]
        
        print(f"\n[Top-5 예측 결과]")
        for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
            label = categories[idx.item()] if idx.item() < len(categories) else f"class_{idx.item()}"
            print(f"  {i+1}. {label}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
            
    except Exception as e:
        print(f"\n[Top-5 예측 결과 (인덱스)]")
        for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
            print(f"  {i+1}. class_{idx.item()}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
    
    return top5_probs[0], top5_indices[0]

if __name__=='__main__':
    use_timm_vit()