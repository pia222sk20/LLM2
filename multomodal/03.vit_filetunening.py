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
    Resize,
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
dataset = load_dataset('imagefolder', data_dir=path, split='train[:1000]')  # 로컬데이터를 사용 
# dataset = load_dataset('food101', data_dir=path, split='train[:1000]')  # 허깅페이스 데이터셋 사용
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
train_transforms =  Compose([
    # RandomResizedCrop(size),
    Resize(256),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    normalize
])
# 검증용 데이터 변환(데이터 증강 없음)
val_transforms =  Compose([
    # RandomResizedCrop(size),    
    Resize(256),
    ToTensor(),
    normalize
])

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

train_dataset = dataset['train'].with_transform(process_train)
test_dataset = dataset['test'].with_transform(process_val)

print('\n데이터증강 파이프라인 설정 완료')

print('\n모델 로드 중 ....')

# 라벨 매핑 생성
labels = selected_classes
label2id =  {label : i  for i, label in enumerate(labels)}
id2label = {i : label  for i, label in enumerate(labels)}

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels = len(labels),
    id2label = label2id,
    label2id = label2id,
    ignore_mismatched_sizes = True  # 분류헤더의 크기가 불일치 무시
)

# 평가 메트릭 정의
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions,axis=1)
    accuracy =  accuracy_score(labels,predictions)
    precision, recall, f1, _ =  precision_recall_fscore_support(
        labels,predictions,average='weighted'
    )
    return {
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall,
        'f1' : f1
    }

# 학습설정
training_args =  TrainingArguments(
    output_dir='./vit_finetuned_food101',
    remove_unused_columns=False,
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate= 5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end= True,
    metric_for_best_model='accuracy',
    logging_dir='./log',
    save_total_limit=2,
    seed=42
)
# Trainer 생성 및 학습
trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics= compute_metrics
)
print('\n학습을 시작합니다.')
try:
    train_results = trainer.train()
    # 최종평가
    eval_results =  trainer.evaluate()
    print('학습완료')
    print(f"정확도(accuracy) : {eval_results['eval_accuracy']:4.f}")
    print(f"정밀도(Precision) : {eval_results['eval_precision']:4.f}")
    print(f"재현율(Recall) : {eval_results['eval_recall']:4.f}")
    print(f"F1 점수(F1) : {eval_results['eval_f1']:4.f}")
    # 모델 저장
    trainer.save_model('./vit_finetuned_food101_final')
    print(f"모델 저장완료 : ./vit_finetuned_food101_final")
except Exception as e:
    print(f'error : {e}')