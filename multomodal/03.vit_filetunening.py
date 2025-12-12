import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from torchvision.transforms import (
    RandomResizedCrop,
    Compose,
    Normalize,
    ToTensor,
    RandomHorizontalFlip
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 데이터셋 로드
# Food-101 5개 클래스만 선택
selected_classes = ['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare']
path = r'C:\Users\Playdata2\.cache\kagglehub\datasets\dansbecker\food-101\versions\1\food-101\food-101\images'
dataset = load_dataset(path, split='train[:1000]')
print(dataset)

# 선택한 클래스만 필터링
def filter_classes(dataset):
    return dataset['label'] in range(len(selected_classes))

dataset = dataset.filter(filter_classes)
dataset = dataset.train_test_split(test_size=0.2,seed=42)

print(f"훈련데이터 : {len(dataset['train'])}개")
print(f"테스트 데이터 : {len(dataset['test'])}개")
print(f"클래스 : {selected_classes}")

# 이미지 프로세스 로드
checkpoint = 'google/vit-base-patch16-224'
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
print(f'이미지 크기 : {image_processor.size}')
print(f'정규화 mean : {image_processor.image_mean}')
print(f'정규화 std : {image_processor.image_std}')
# 데이터증강  - 파이프라인
normalize = Normalize(
    mean = image_processor.image_mean,
    std = image_processor.image_std
)
size = ( image_processor.size['shortest_edge']  
        if 'shortest_edge' in image_processor.size  
        else (image_processor.size['height'], image_processor.size['width'])
        )
# 훈련용 데이터 변환(데이터 증강 포함)
train_transforms =  Compose(
    RandomResizedCrop(size),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    normalize
)
# 검증용 데이터 변환(데이터 증강 없음)
val_transforms =  Compose(
    RandomResizedCrop(size),    
    ToTensor(),
    normalize
)

# 변환적용
def process_train(examples):
    examples['pixel_values'] = [
        train_transforms(img.convert('RGB')) for img in examples['image']
    ]
    return examples

def process_val(examples):
    examples['pixel_values'] = [
        val_transforms(img.convert('RGB')) for img in examples['image']
    ]
    return examples

train_dataset = dataset['train'].with_transform()
test_dataset = dataset['test'].with_transform()