# 벡터 데이터베이스 (Vector Databases)

## 개요

대규모 벡터 데이터를 효율적으로 저장하고 검색하는 방법을 학습합니다.
- FAISS 인덱스 종류와 특성
- 양자화 기법 (Product Quantization)
- 근사 최근접 이웃 검색 (ANN)

## 핵심 알고리즘

### 1. 정확한 검색 vs 근사 검색

**정확한 검색 (Exact Search)**:
- 모든 벡터와 비교
- 시간 복잡도: O(n × d) (n: 문서 수, d: 차원)
- 100% 정확하지만 느림

**근사 검색 (Approximate Nearest Neighbor, ANN)**:
- 일부 벡터만 비교
- 시간 복잡도: O(log n) ~ O(√n)
- 약간의 정확도 손실, 매우 빠름

### 2. FAISS 인덱스 종류

| 인덱스 | 설명 | 메모리 | 속도 | 정확도 |
|--------|------|--------|------|--------|
| `IndexFlatL2` | Brute-force L2 | 높음 | 느림 | 100% |
| `IndexFlatIP` | Brute-force 내적 | 높음 | 느림 | 100% |
| `IndexIVFFlat` | IVF + Flat | 높음 | 빠름 | 높음 |
| `IndexIVFPQ` | IVF + PQ | **낮음** | 빠름 | 중간 |
| `IndexHNSWFlat` | HNSW 그래프 | 중간 | 매우 빠름 | 높음 |

### 3. IVF (Inverted File Index)

**원리**: 벡터 공간을 여러 클러스터로 분할하고, 검색 시 관련 클러스터만 탐색

```
1. 학습 단계: K-means로 nlist개의 클러스터 생성
2. 추가 단계: 각 벡터를 가장 가까운 클러스터에 할당
3. 검색 단계: 쿼리와 가장 가까운 nprobe개 클러스터만 탐색
```

**파라미터**:
- `nlist`: 클러스터 수 (보통 √n ~ 4√n)
- `nprobe`: 검색할 클러스터 수 (클수록 정확, 느림)

### 4. Product Quantization (PQ)

**원리**: 고차원 벡터를 여러 서브벡터로 나누고 각각을 양자화

```
768차원 벡터 → 8개의 96차원 서브벡터 → 각각 256개 클러스터로 양자화

원래 크기: 768 × 4 bytes = 3,072 bytes
PQ 후: 8 × 1 byte = 8 bytes (384배 압축!)
```

**장점**:
- 메모리 대폭 절감
- 코드북 기반 빠른 거리 계산

**단점**:
- 정확도 손실
- 학습 시간 필요

### 5. HNSW (Hierarchical Navigable Small World)

**원리**: 다층 그래프 구조로 효율적인 탐색

```
Layer 2:  [A] -------- [F]
           |            |
Layer 1:  [A] -- [C] -- [F]
           |    / \     |
Layer 0:  [A]-[B]-[C]-[D]-[E]-[F]
```

**특징**:
- 상위 레이어: 장거리 점프 (빠른 탐색)
- 하위 레이어: 지역 탐색 (정밀도)
- 검색 시간: O(log n)
