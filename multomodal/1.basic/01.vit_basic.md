# Vision Transformer (ViT) 기초 이론

## 1. ViT란 무엇인가?

### 1.1 개요
Vision Transformer(ViT)는 2020년 Google Research에서 발표한 논문 "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"에서 소개된 모델입니다.

**핵심 아이디어**: 이미지를 시퀀스처럼 처리하여 NLP에서 성공한 Transformer 아키텍처를 컴퓨터 비전에 적용

### 1.2 기존 CNN과의 차이점

| 특성 | CNN | ViT |
|-----|-----|-----|
| 연산 방식 | 지역적 컨볼루션 | 전역적 Self-Attention |
| Inductive Bias | 강함 (locality, translation equivariance) | 약함 |
| 데이터 효율성 | 적은 데이터로도 학습 가능 | 대용량 데이터 필요 |
| 장거리 의존성 | 레이어 쌓아야 포착 | 첫 레이어부터 포착 가능 |
| 해석 가능성 | CAM, Grad-CAM | Attention Map |

---

## 2. ViT 아키텍처

### 2.1 전체 구조

ViT의 전체 처리 과정은 다음과 같습니다. 이미지를 패치로 자르고, 이를 일렬로 나열한 뒤 Transformer에 입력합니다.

![vit overall Architecture](./img/vit%20overall%20Architecture.png)

**처리 흐름**:
1. **입력 이미지**: 224×224×3 크기의 RGB 이미지
2. **패치 분할**: 16×16 패치 196개로 분할
3. **선형 투영**: 각 패치를 768차원 벡터로 변환
4. **CLS 토큰**: 분류를 위한 특별 토큰 추가
5. **위치 임베딩**: 순서 정보 추가
6. **Transformer Encoder**: L개의 블록 통과
7. **분류 헤드**: CLS 토큰으로 최종 분류

### 2.2 패치 임베딩 (Patch Embedding)

이미지를 고정 크기의 패치로 분할하고, 각 패치를 벡터로 변환합니다. 마치 문장을 단어(토큰)로 나누는 것과 같습니다.

![Patch Embedding Process](./img/Patch%20Embedding%20Process.png)

**핵심 포인트**:
- 각 16×16 패치는 하나의 "토큰"처럼 처리됨
- NLP의 단어 임베딩과 동일한 역할
- Conv2d(kernel=16, stride=16)로 효율적 구현 가능

**수식**:
- 이미지 크기: $H \times W \times C$ (예: 224 x 224 x 3)
- 패치 크기: $P \times P$ (예: 16 x 16)
- 패치 수: $N = \frac{H \times W}{P^2}$ (예: 196개)
- 패치 임베딩: $x_p^i E$ where $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$

```python
# 패치 임베딩 예시
patch_size = 16
image_size = 224
num_patches = (image_size // patch_size) ** 2  # 196
patch_dim = patch_size * patch_size * 3        # 768 (16*16*3)
embedding_dim = 768                            # 임베딩 차원
```

### 2.3 위치 임베딩 (Positional Embedding)

Transformer는 순서 정보가 없으므로, 위치 정보를 추가합니다.

**ViT의 위치 임베딩 특징**:
- 학습 가능한 1D 위치 임베딩 사용
- 2D 위치 임베딩보다 성능 차이 미미
- 각 패치 위치에 고유한 임베딩 벡터 추가

```python
# 위치 임베딩
position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
# +1은 CLS 토큰을 위한 것
```

### 2.4 CLS 토큰

BERT에서 영감을 받은 특별한 토큰으로, 전체 이미지의 표현을 학습합니다.

```python
# CLS 토큰
cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
# 최종 분류는 CLS 토큰의 출력을 사용
```

### 2.5 Transformer Encoder

Transformer Encoder는 입력된 패치 간의 관계를 학습하는 핵심 모듈입니다.

![Transformer Encoder Block](./img/Transformer%20Encoder%20Block.png)

**구성 요소**:
- **Layer Normalization**: 학습 안정화
- **Multi-Head Self-Attention**: 패치 간 관계 학습
- **MLP**: 비선형 변환 (차원 확장 후 축소)
- **Residual Connection**: 그래디언트 흐름 개선

#### Multi-Head Self-Attention (MSA)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Query, Key, Value**: 입력으로부터 선형 변환으로 생성
- **Multi-Head**: 여러 개의 attention을 병렬로 수행
- **Self-Attention**: Q, K, V가 모두 같은 입력에서 생성

```python
# Multi-Head Attention 예시
num_heads = 12
head_dim = embedding_dim // num_heads  # 64

# Attention 계산
attention = softmax(Q @ K.T / sqrt(head_dim)) @ V
```

#### MLP (Feed-Forward Network)

$$\text{MLP}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

- 일반적으로 hidden dimension은 4배로 확장
- GELU 활성화 함수 사용

```python
# MLP 예시
mlp_dim = embedding_dim * 4  # 3072
# Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768)
```

---

## 3. ViT 모델 변형

### 3.1 기본 모델 크기

| 모델 | Layers | Hidden Size | MLP Size | Heads | Params |
|------|--------|-------------|----------|-------|--------|
| ViT-Tiny | 12 | 192 | 768 | 3 | 5.7M |
| ViT-Small | 12 | 384 | 1536 | 6 | 22M |
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

### 3.2 패치 크기에 따른 명명 규칙

- **ViT-B/16**: Base 모델, 16x16 패치
- **ViT-B/32**: Base 모델, 32x32 패치
- **ViT-L/14**: Large 모델, 14x14 패치

패치 크기가 작을수록:
- 더 많은 패치 생성 -> 더 많은 계산
- 더 세밀한 특징 포착 가능
- 일반적으로 더 좋은 성능

---

## 4. ViT의 학습

### 4.1 사전 학습 (Pre-training)

**데이터셋**:
- ImageNet-21k (14M 이미지, 21k 클래스)
- JFT-300M (Google 내부 데이터셋, 300M 이미지)

**학습 설정**:
- Optimizer: Adam (beta1=0.9, beta2=0.999)
- Learning Rate: Warmup + Linear/Cosine Decay
- Batch Size: 4096
- Data Augmentation: RandAugment, Mixup, Cutmix

### 4.2 파인튜닝 (Fine-tuning)

사전 학습된 모델을 특정 태스크에 맞게 추가 학습:

1. **Higher Resolution**: 사전학습보다 높은 해상도 사용
2. **Position Embedding Interpolation**: 해상도 변경 시 위치 임베딩 보간
3. **Smaller Learning Rate**: 1e-5 ~ 1e-4 정도

```python
# 파인튜닝 예시 설정
learning_rate = 3e-5
batch_size = 32
epochs = 10
```

---

## 5. ViT의 특징

### 5.1 장점

1. **전역적 특징 학습**: Self-Attention으로 이미지 전체 관계 포착
2. **확장성**: 모델/데이터 크기에 따른 성능 향상 예측 가능
3. **전이학습 우수**: 대규모 사전학습 후 다양한 태스크에 적용
4. **해석 가능성**: Attention Map으로 모델 판단 근거 시각화

### 5.2 단점

1. **데이터 요구량**: 적은 데이터에서 CNN보다 성능 저하
2. **계산 비용**: Self-Attention의 O(n^2) 복잡도
3. **Inductive Bias 부족**: 이미지 특성 반영 어려움

### 5.3 Attention Map 시각화

```python
# Attention Map 추출 예시
attention_weights = model.get_attention_weights()
# shape: (batch, heads, num_patches+1, num_patches+1)

# CLS 토큰의 다른 패치에 대한 attention 시각화
cls_attention = attention_weights[:, :, 0, 1:]  # CLS -> 패치들
```

---

## 6. ViT 변형 모델들

### 6.1 DeiT (Data-efficient Image Transformer)

- Facebook AI에서 개발
- ImageNet-1k만으로 학습 가능
- Knowledge Distillation 적용
- Distillation Token 도입

### 6.2 Swin Transformer

- Microsoft에서 개발
- 계층적 구조 (Hierarchical)
- Shifted Window Attention
- 다양한 비전 태스크에 적용 가능

### 6.3 BEiT (BERT Pre-training of Image Transformers)

- Masked Image Modeling
- BERT 스타일의 사전학습
- Visual Token 예측

### 6.4 CLIP (Contrastive Language-Image Pre-training)

- OpenAI에서 개발
- 이미지-텍스트 쌍으로 학습
- Zero-shot 분류 가능
- 멀티모달 응용

---

## 7. 실습 준비

### 7.1 필요한 라이브러리

```python
# 핵심 라이브러리
import torch
import torchvision
from transformers import ViTModel, ViTForImageClassification
import timm

# 보조 라이브러리
from PIL import Image
import matplotlib.pyplot as plt
```

### 7.2 사전학습 모델 사용

```python
# Hugging Face Transformers
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# timm
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True)
```
## 8. 참고 자료

### 논문
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - ViT 원논문
- [DeiT](https://arxiv.org/abs/2012.12877) - Data-efficient Image Transformer
- [Swin Transformer](https://arxiv.org/abs/2103.14030) - Hierarchical Vision Transformer
- [CLIP](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models

### 코드
- [Google ViT 공식 구현](https://github.com/google-research/vision_transformer)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/vit)
- [timm 라이브러리](https://github.com/huggingface/pytorch-image-models)

### 튜토리얼
- [Hugging Face ViT 튜토리얼](https://huggingface.co/docs/transformers/tasks/image_classification)
- [PyTorch Image Models 문서](https://timm.fast.ai/)
