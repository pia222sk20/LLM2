# vit pipeline
import torch
from transformers import pipeline
classifier = pipeline(
    task = 'image-classification',
    model = 'google/vit-base-patch16-224',
    device = 0 if torch.cuda.is_available() else -1
)
test_images = [
    {
        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "description": "고양이 이미지"
    },
    {
        "url": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "description": "고양이 2마리 이미지"
    },
    {
        "url": "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
        "description": "앵무새 이미지"
    }
]
for idx, img_info in enumerate(test_images, 1):
    print(f" -- {idx} : {img_info['description']} --")
    try:
        results = classifier(img_info['url'])  # label  score
        print(f'예측결과 : {len(results)}')
        for idx, result in enumerate(results,1):
            print(f"    {idx}  label:{result['label']}  score : {result['score']:.2f} ")
    except Exception as e:
        print(f'error : {e}')