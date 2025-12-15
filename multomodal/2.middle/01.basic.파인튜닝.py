import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import argparse
import sys

# 상위 디렉토리의 config 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_device, set_seed, get_optimal_batch_size


def print_section(title):
    """섹션 제목 출력"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


# ============================================================
# 1. 환경 설정
# ============================================================
def setup_environment(force_cpu=False):
    """환경 설정"""
    print_section("1. 환경 설정")
    
    # 디바이스 설정 (config 모듈 사용)
    device = get_device(force_cpu=force_cpu)
    
    # 재현성을 위한 시드 설정
    set_seed(42)
    
    print(f"\n[시드 설정]")
    print(f"  랜덤 시드: 42")
    
    return device


# ============================================================
# 2. 데이터셋 준비 (CIFAR-10)
# ============================================================
def prepare_cifar10_dataset():
    """CIFAR-10 데이터셋 준비"""
    print_section("2. CIFAR-10 데이터셋 준비")
    
    # CIFAR-10 클래스
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"\n[CIFAR-10 정보]")
    print(f"  클래스 수: 10")
    print(f"  클래스: {classes}")
    print(f"  이미지 크기: 32 x 32")
    print(f"  학습 데이터: 50,000장")
    print(f"  테스트 데이터: 10,000장")
    
    # ImageNet 통계 (ViT 사전학습에 사용됨)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # ViT는 224x224 입력이 필요하므로 리사이즈
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    print(f"\n[데이터 증강]")
    print(f"  학습용: Resize, RandomHorizontalFlip, RandomRotation, ColorJitter")
    print(f"  검증용: Resize만 적용")
    
    # 데이터셋 로드
    print(f"\n[데이터셋 다운로드/로드 중...]")
    data_dir = './data'
    
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )
    
    # 학습/검증 분리 (학습의 10%를 검증으로)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"\n[데이터 분할]")
    print(f"  학습 데이터: {len(train_dataset)}")
    print(f"  검증 데이터: {len(val_dataset)}")
    print(f"  테스트 데이터: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, classes


# ============================================================
# 3. 데이터 로더 생성
# ============================================================
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """데이터 로더 생성"""
    print_section("3. 데이터 로더 생성")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    print(f"\n[데이터 로더 설정]")
    print(f"  배치 크기: {batch_size}")
    print(f"  학습 배치 수: {len(train_loader)}")
    print(f"  검증 배치 수: {len(val_loader)}")
    print(f"  테스트 배치 수: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# ============================================================
# 4. ViT 모델 준비
# ============================================================
def prepare_vit_model(num_classes, device, strategy='full'):
    """ViT 모델 준비"""
    print_section("4. ViT 모델 준비")
    
    try:
        from transformers import ViTForImageClassification, ViTConfig
        
        # 사전학습 모델 로드
        model_name = "google/vit-base-patch16-224"
        print(f"\n[모델 로드]")
        print(f"  사전학습 모델: {model_name}")
        
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # classifier 크기 불일치 무시
        )
        
        print(f"\n[원본 모델 정보]")
        print(f"  원본 클래스 수: 1000 (ImageNet)")
        print(f"  새 클래스 수: {num_classes}")
        
        # 파인튜닝 전략 적용
        if strategy == 'linear_probe':
            # Linear Probing: Encoder 동결
            print(f"\n[파인튜닝 전략: Linear Probing]")
            for param in model.vit.parameters():
                param.requires_grad = False
            print(f"  ViT Encoder: 동결됨 (학습 안 함)")
            print(f"  Classifier: 학습됨")
            
        elif strategy == 'partial':
            # 일부 레이어만 학습 (마지막 2개 블록)
            print(f"\n[파인튜닝 전략: Partial Fine-tuning]")
            for param in model.vit.parameters():
                param.requires_grad = False
            # 마지막 2개 encoder 블록 학습
            for param in model.vit.encoder.layer[-2:].parameters():
                param.requires_grad = True
            print(f"  ViT Encoder (Layer 0-9): 동결됨")
            print(f"  ViT Encoder (Layer 10-11): 학습됨")
            print(f"  Classifier: 학습됨")
            
        else:  # full
            print(f"\n[파인튜닝 전략: Full Fine-tuning]")
            print(f"  전체 모델 학습")
        
        # 학습 가능한 파라미터 수 계산
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n[파라미터 정보]")
        print(f"  전체 파라미터: {total_params:,}")
        print(f"  학습 파라미터: {trainable_params:,}")
        print(f"  학습 비율: {trainable_params/total_params*100:.2f}%")
        
        model = model.to(device)
        
        return model
        
    except ImportError:
        print("  transformers 라이브러리가 필요합니다.")
        return None


# ============================================================
# 5. 학습 함수
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """한 에포크 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, predicted = outputs.logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 진행 상황 업데이트
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """모델 평가"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


# ============================================================
# 6. 전체 학습 루프
# ============================================================
def train_model(model, train_loader, val_loader, device, num_epochs=5, lr=1e-4):
    """전체 학습 루프"""
    print_section("6. 모델 학습")
    
    if model is None:
        print("  모델이 준비되지 않았습니다.")
        return None
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    
    # Learning rate 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    print(f"\n[학습 설정]")
    print(f"  손실 함수: CrossEntropyLoss")
    print(f"  옵티마이저: AdamW")
    print(f"  초기 Learning Rate: {lr}")
    print(f"  Weight Decay: 0.01")
    print(f"  스케줄러: CosineAnnealingLR")
    print(f"  에포크 수: {num_epochs}")
    
    # 학습 기록
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"\n[학습 시작]")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # 학습
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 검증
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 기록 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 출력
        print(f"\n  Epoch {epoch}/{num_epochs}")
        print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"    LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"    >> Best model saved! (Val Acc: {best_val_acc:.2f}%)")
    
    elapsed_time = time.time() - start_time
    print(f"\n[학습 완료]")
    print(f"  총 학습 시간: {elapsed_time/60:.2f}분")
    print(f"  최고 검증 정확도: {best_val_acc:.2f}%")
    
    return history


# ============================================================
# 7. 학습 결과 시각화
# ============================================================
def plot_training_history(history):
    """학습 기록 시각화"""
    print_section("7. 학습 결과 시각화")
    
    if history is None:
        print("  학습 기록이 없습니다.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 그래프
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy 그래프
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    save_path = 'training_history.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  학습 기록 저장: {save_path}")
    
    plt.show()


# ============================================================
# 8. 테스트 및 혼동 행렬
# ============================================================
def test_model(model, test_loader, device, classes):
    """테스트 세트에서 모델 평가"""
    print_section("8. 테스트 평가")
    
    if model is None:
        print("  모델이 준비되지 않았습니다.")
        return
    
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 전체 정확도
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100. * correct / len(all_labels)
    
    print(f"\n[테스트 결과]")
    print(f"  테스트 정확도: {accuracy:.2f}%")
    
    # 클래스별 정확도
    print(f"\n[클래스별 정확도]")
    for i, cls in enumerate(classes):
        cls_indices = [j for j, l in enumerate(all_labels) if l == i]
        cls_correct = sum(all_preds[j] == i for j in cls_indices)
        cls_acc = 100. * cls_correct / len(cls_indices) if cls_indices else 0
        print(f"  {cls:<12}: {cls_acc:.2f}%")
    
    # 혼동 행렬 시각화
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        save_path = 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  혼동 행렬 저장: {save_path}")
        
        plt.show()
        
    except ImportError:
        print("\n  sklearn, seaborn 라이브러리가 필요합니다.")


# ============================================================
# 9. 빠른 테스트 (데모용)
# ============================================================
def quick_demo(device):
    """빠른 데모 (적은 데이터로 테스트)"""
    print_section("빠른 데모 모드")
    
    print("\n[데모 모드로 실행]")
    print("  - 데이터 1000장만 사용")
    print("  - 1 에포크만 학습")
    print("  - Linear Probing 전략")
    
    # 데이터셋 준비
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    # CIFAR-10 로드
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # 1000장만 사용
    indices = list(range(1000))
    train_subset = torch.utils.data.Subset(train_dataset, indices[:800])
    val_subset = torch.utils.data.Subset(train_dataset, indices[800:])
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
    
    # 모델 준비 (Linear Probing)
    model = prepare_vit_model(num_classes=10, device=device, strategy='linear_probe')
    
    if model is None:
        return
    
    # 1 에포크 학습
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    
    print("\n[학습 시작 (1 에포크)]")
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device, 1
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    print(f"\n[결과]")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")


# ============================================================
# 메인 실행
# ============================================================
def main(force_cpu=False):
    """메인 실행 함수"""
    print("\n" + "#" * 60)
    print("# ViT 파인튜닝 기초 실습")
    print("#" * 60)
    
    # 환경 설정
    device = setup_environment(force_cpu=force_cpu)
    
    # 사용자 선택
    print("\n[실행 모드 선택]")
    print("  1. 빠른 데모 (1000장, 1 에포크)")
    print("  2. 전체 학습 (50000장, 5 에포크)")
    
    # 데모 모드로 실행 (실제 사용 시 input()으로 선택)
    mode = "1"  # 데모 모드
    
    if mode == "1":
        quick_demo(device)
    else:
        # 데이터셋 준비
        train_dataset, val_dataset, test_dataset, classes = prepare_cifar10_dataset()
        
        # 데이터 로더 생성
        batch_size = get_optimal_batch_size('base', device)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, batch_size=batch_size
        )
        
        # 모델 준비
        model = prepare_vit_model(num_classes=10, device=device, strategy='linear_probe')
        
        # 학습
        history = train_model(model, train_loader, val_loader, device, 
                             num_epochs=5, lr=1e-3)
        
        # 결과 시각화
        plot_training_history(history)
        
        # 테스트
        test_model(model, test_loader, device, classes)