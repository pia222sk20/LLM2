import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from torchvision.transforms import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print("데이터셋 로드 중...")

# train 전체 로드
dataset = load_dataset('food101', split='train')
test_dataset_full = load_dataset('food101', split='validation')

selected_classes = [
    'apple_pie',
    'baby_back_ribs',
    'baklava',
    'beef_carpaccio',
    'beef_tartare'
]

label_names = dataset.features["label"].names
selected_ids = [label_names.index(cls) for cls in selected_classes]

print("선택 클래스 인덱스 =", selected_ids)

def filter_class(example):
    return example["label"] in selected_ids

dataset = dataset.filter(filter_class)
test_dataset_full = test_dataset_full.filter(filter_class)

dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = dataset['train']
test_dataset = dataset['test']

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

if len(train_dataset) == 0:
    raise ValueError("필터링 후 데이터가 0개입니다.")

checkpoint = 'google/vit-base-patch16-224'
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(
    mean=image_processor.image_mean,
    std=image_processor.image_std
)

train_transforms = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize
])

val_transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    normalize
])

def process_train(examples):
    examples["pixel_values"] = [train_transforms(img.convert("RGB")) for img in examples["image"]]
    return examples

def process_val(examples):
    examples["pixel_values"] = [val_transforms(img.convert("RGB")) for img in examples["image"]]
    return examples

train_dataset = train_dataset.with_transform(process_train)
test_dataset = test_dataset.with_transform(process_val)

label2id = {label: i for i, label in enumerate(selected_classes)}
id2label = {i: label for i, label in enumerate(selected_classes)}

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(selected_classes),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}

training_args = TrainingArguments(
    output_dir='./vit_finetuned_food101',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    load_best_model_at_end=True,
    logging_dir='./log'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
