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

model = AutoModelForImageClassification.from_pretrained(
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
        inputs = image_processor(image, return_tensors='pt').to(model.device)
        print(f"전처리 후 텐서의 크기 : {inputs['pixel_values'].shape}")
        # 추론
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        # 결과 해석
        predicted_class_id =  logits.argmax(dim=-1).item()
        predicted_class_label = model.config.id2label[predicted_class_id]
        confidence = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()
        print('예측결과: ....')
        print(f'클래스 ID: {predicted_class_id}')
        print(f'클래스 라벨: {predicted_class_label}')
        print(f'확률(신뢰도): {confidence}')
        # top5 예측 결과
        probs = torch.softmax(logits, dim=-1)[0]
        top5_probs, top5_indices = torch.topk(probs, 5)
        for i, (prob, idx) in enumerate(zip(top5_probs,top5_indices),1):
            label = model.config.id2label[idx.item()]
            print(f'    {i}. {label} : {prob.item():.2f}')
    except Exception as e:
        print(f'오류발생 : {e}')