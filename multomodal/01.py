import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
# 모델 이미지 프로세스 로드
model_name = 'google/vit-base-patch16-224'
image_processor =  AutoImageProcessor.from_pretrained(
    model_name,    
    use_fast=True
)

model = AutoModelForImageClassification(
    model_name,    
    device_map = 'auto'
)
# 이미지 로드
image_urls = [
"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
"http://images.cocodataset.org/val2017/000000039769.jpg",
]

for idx, url in enumerate(image_urls,1):
    try:
        # 이미지 다운로드
        image =  Image.open( requests.get(url, stream=True).raw )
        print(f'이미지 크기 : {image.size}')
        # 이미지 전처리
        inputs = image_processor(image, return_tensor='pt').to(model.device)
        print(f"전처리 후 텐서의 크기 : {inputs['pixel_values'].shape}")
        # 추론
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        # 결과 해석
        predicted_class_id =  logits.argmax(dim=-1).item()
        predicted_class_label = model.config.id2label[predicted_class_id]
        