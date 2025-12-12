"""
공개 데이터셋(Food-101)을 사용한 ViT 파인튜닝
"""

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

def main():
    print("=" * 60)
    print("Vision Transformer (ViT) Fine-tuning 예제")
    print("=" * 60)
    
    # 1. 데이터셋 로드 (Food-101의 일부만 사용)
    print("\n[1단계] 데이터셋 로드 중...")
    
    # Food-101 데이터셋에서 5개 클래스만 선택 
    selected_classes = ["apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare"]  # [0 1 2 3 4]
    
    dataset = load_dataset("food101", split="train") 
    
    print(" 데이터 필터링 시작 (잠시만 기다려주세요)...")
    # 선택한 클래스만 필터링
    def filter_classes(example):
        return example['label'] in range(len(selected_classes))
    
    dataset = dataset.filter(filter_classes)
    
    # if len(dataset) > 1000:
    #     dataset = dataset.shuffle(seed=42).select(range(1000))

    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    print(f" 훈련 데이터: {len(dataset['train'])}개")
    print(f" 테스트 데이터: {len(dataset['test'])}개")
    print(f" 클래스: {selected_classes}")
    
    # 2. 이미지 프로세서 로드
    print("\n[2단계] 이미지 프로세서 로드 중...")
    
    checkpoint = "google/vit-base-patch16-224"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    
    print(f" 이미지 크기: {image_processor.size}")
    print(f" 정규화 mean: {image_processor.image_mean}")
    print(f" 정규화 std: {image_processor.image_std}")
    
    # 3. 데이터 증강 설정
    print("\n[3단계] 데이터 증강 파이프라인 설정 중...")
    
    normalize = Normalize(
        mean=image_processor.image_mean,
        std=image_processor.image_std
    )
    
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    
    # 훈련용 변환 (데이터 증강 포함)
    train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        normalize,
    ])
    
    # 검증용 변환 (데이터 증강 없음)
    val_transforms = Compose([
        RandomResizedCrop(size),
        ToTensor(),
        normalize,
    ])
    
    def preprocess_train(examples):
        examples["pixel_values"] = [
            train_transforms(img.convert("RGB")) for img in examples["image"]
        ]
        return examples
    
    def preprocess_val(examples):
        examples["pixel_values"] = [
            val_transforms(img.convert("RGB")) for img in examples["image"]
        ]
        return examples
    
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['label'] for x in batch])
        }    
    
    # 변환 적용
    train_dataset = dataset["train"].with_transform(preprocess_train)
    test_dataset = dataset["test"].with_transform(preprocess_val)
    
    print(" 데이터 증강 파이프라인 설정 완료")
    
    # 4. 모델 로드
    print("\n[4단계] 모델 로드 중...")
    
    # 라벨 매핑 생성
    labels = selected_classes
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # 분류 헤드 크기 불일치 무시
    )
    
    print(f" 모델 로드 완료")
    print(f" 클래스 수: {len(labels)}")
    print(f" 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 평가 메트릭 정의
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    # 6. 학습 설정
    print("\n[5단계] 학습 설정 중...")
    
    training_args = TrainingArguments(
        output_dir="./vit_finetuned_food101",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        seed=42,
    )
    
    print(f"✓ 학습률: {training_args.learning_rate}")
    print(f"✓ 배치 크기: {training_args.per_device_train_batch_size}")
    print(f"✓ 에폭 수: {training_args.num_train_epochs}")
    
    # 7. Trainer 생성 및 학습
    print("\n[6단계] Trainer 생성 및 학습 시작...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn  # 허깅페이스 Trainer api를 이용할때 반드시 적용(학습에 필요한 데이터만 처리할수 있도록)
    )
    
    print("\n학습을 시작합니다...")    
    
    try:
        train_results = trainer.train()
        
        # 8. 최종 평가
        print("\n[7단계] 최종 평가 수행 중...")
        eval_results = trainer.evaluate()
        
        print("\n" + "=" * 60)
        print("학습 완료!")
        print("=" * 60)
        print(f"\n최종 결과:")
        print(f"  정확도 (Accuracy): {eval_results['eval_accuracy']:.4f}")
        print(f"  정밀도 (Precision): {eval_results['eval_precision']:.4f}")
        print(f"  재현율 (Recall): {eval_results['eval_recall']:.4f}")
        print(f"  F1 점수: {eval_results['eval_f1']:.4f}")
        
        # 9. 모델 저장
        print("\n[8단계] 모델 저장 중...")
        trainer.save_model("./vit_finetuned_food101_final")
        print("모델 저장 완료: ./vit_finetuned_food101_final")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")        
    
    print("\n" + "=" * 60)
    print("Fine-tuning 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()