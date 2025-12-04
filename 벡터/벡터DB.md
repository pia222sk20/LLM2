# 임베딩과 VectorDB 이론

## 목차

1. [임베딩(Embedding)의 개념](#1-임베딩embedding의-개념)
2. [임베딩 모델 비교](#2-임베딩-모델-비교)
3. [유사도 측정 방법](#3-유사도-측정-방법)
4. [VectorDB 개요](#4-vectordb-개요)
5. [ChromaDB 심화](#5-chromadb-심화)
6. [적용 가이드](#6-적용-가이드)

---

## 1. 임베딩(Embedding)의 개념

### 1.1 임베딩이란?

**임베딩(Embedding)**은 텍스트, 이미지 등의 데이터를 **고정 길이의 숫자 벡터**로 변환하는 기술입니다.

```text
"고양이는 귀엽다" → [0.23, -0.45, 0.12, 0.89, ..., 0.78]
                           ↑
                    1536차원의 벡터 (OpenAI)
```

### 1.2 임베딩의 핵심 원리

임베딩의 가장 중요한 특성은 **의미적 유사성을 거리로 표현**한다는 것입니다.

```text
벡터 공간에서의 거리:

    "고양이" ●───────● "강아지"     (가까움: 유사한 의미)
              \
               \
                \
                 ● "자동차"         (멀리 떨어짐: 다른 의미)
```

### 1.3 임베딩이 가능한 이유

딥러닝 모델(Transformer)이 대량의 텍스트를 학습하면서:

1. **문맥 이해**: "왕 - 남자 + 여자 ≈ 여왕" 같은 관계 학습
2. **의미 압축**: 텍스트의 핵심 의미를 벡터로 압축
3. **유사성 보존**: 비슷한 의미의 텍스트는 비슷한 벡터

### 1.4 임베딩 차원(Dimension)의 의미

| 차원 | 특징 | 예시 모델 |
|------|------|-----------|
| 낮은 차원 (384) | 빠름, 저장 효율적 | all-MiniLM-L6 |
| 중간 차원 (1024) | 균형 잡힌 선택 | BGE-M3, E5 |
| 높은 차원 (1536+) | 높은 표현력 | OpenAI text-embedding-3 |

> **차원이 높을수록**: 더 미세한 의미 차이를 표현할 수 있지만, 저장 공간과 계산 비용 증가

---

## 2. 임베딩 모델 비교

### 2.1 주요 임베딩 모델

| 모델 | 차원 | 한국어 | 비용 | 특징 |
|------|------|--------|------|------|
| **text-embedding-3-small** | 1536 | △ | 저렴 | OpenAI, 범용 |
| **text-embedding-3-large** | 3072 | △ | 높음 | OpenAI, 고품질 |
| **text-embedding-ada-002** | 1536 | △ | 중간 | 이전 세대 |
| **BGE-M3** | 1024 | ◎ | 무료 | 다국어 특화, 한국어 우수 |
| **multilingual-e5-large** | 1024 | ○ | 무료 | 다국어 지원 |
| **KoSimCSE** | 768 | ◎ | 무료 | 한국어 전용 |

### 2.2 모델 선택 가이드

```text
프로젝트 시작 / 영어 중심
    └── OpenAI text-embedding-3-small (권장)

한국어 문서 중심 / 비용 절감
    └── BGE-M3 또는 KoSimCSE

최고 품질 필요 / 프로덕션
    └── text-embedding-3-large
```

### 2.3 OpenAI vs 오픈소스 비교

**OpenAI 임베딩:**

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("안녕하세요")
```

- 장점: 설정 간편, 안정적 성능
- 단점: API 비용 발생, 인터넷 필요

**오픈소스 임베딩 (HuggingFace):**

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"}  # GPU 사용
)
vector = embeddings.embed_query("안녕하세요")
```

- 장점: 무료, 로컬 실행 가능
- 단점: 초기 다운로드 필요, 리소스 사용

---

## 3. 유사도 측정 방법

### 3.1 코사인 유사도 (Cosine Similarity)

가장 널리 사용되는 유사도 측정 방법입니다.

```text
코사인 유사도 = (A · B) / (||A|| × ||B||)

범위: -1 ~ 1
- 1: 완전히 동일한 방향 (유사)
- 0: 직교 (무관)
- -1: 반대 방향 (반대 의미)
```

**수학적 의미:**

```text
        B
       /
      /  θ (각도)
     /
    A ─────────

cos(θ) = 두 벡터 사이 각도의 코사인
각도가 작을수록 (유사할수록) cos 값이 1에 가까움
```

### 3.2 Python 구현

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 예시
similarity = cosine_similarity(embedding1, embedding2)
print(f"유사도: {similarity:.4f}")
```

### 3.3 다른 유사도 측정 방법

| 방법 | 수식 | 특징 |
|------|------|------|
| **코사인 유사도** | cos(θ) | 방향만 비교, 크기 무시 |
| **유클리드 거리** | √Σ(a-b)² | 절대 거리, 값 작을수록 유사 |
| **내적** | A · B | 크기와 방향 모두 고려 |
| **맨해튼 거리** | Σ\|a-b\| | L1 거리 |

> **RAG에서는 코사인 유사도가 표준**: 텍스트 임베딩은 정규화되어 있어 코사인 유사도가 효과적

### 3.4 유사도 실험 예시

```text
기준: "RAG는 검색 증강 생성 기술입니다."

비교 문장                              유사도
─────────────────────────────────────────────
"RAG는 문서 검색과 생성을 결합합니다."    0.85 (높음)
"벡터 데이터베이스는 임베딩 저장합니다."   0.45 (중간)
"오늘 날씨가 좋습니다."                  0.12 (낮음)
```

---

## 4. VectorDB 개요

### 4.1 VectorDB란?

**VectorDB(벡터 데이터베이스)**는 고차원 벡터를 효율적으로 저장하고 검색하는 특수 데이터베이스입니다.

```text
일반 DB:        WHERE name = 'John'    (정확한 매칭)
VectorDB:       SIMILAR TO [0.23, ...]  (유사도 기반 검색)
```

### 4.2 VectorDB가 필요한 이유

**일반 검색 vs 벡터 검색:**

```text
질문: "LLM 애플리케이션 만드는 방법"

키워드 검색:
- "LLM", "애플리케이션", "만드는", "방법" 단어 포함 문서 찾기
-  "대규모 언어 모델로 앱 개발하기" → 단어 불일치로 누락!

벡터 검색:
- 질문을 벡터로 변환 → 유사한 의미의 벡터 찾기
-  "대규모 언어 모델로 앱 개발하기" → 의미 유사로 검색됨!
```

### 4.3 주요 VectorDB 비교

| VectorDB | 유형 | 특징 | 적합한 용도 |
|----------|------|------|-------------|
| **ChromaDB** | 오픈소스 | 파이썬 네이티브, 간편 설치 | 개발, 소규모 프로젝트 |
| **Pinecone** | 클라우드 | 완전 관리형, 확장성 | 프로덕션, 대규모 |
| **Weaviate** | 오픈소스 | 그래프 기반, 하이브리드 검색 | 복잡한 쿼리 |
| **FAISS** | 라이브러리 | Facebook 개발, 고성능 | 대용량, 연구 |
| **Milvus** | 오픈소스 | 분산 처리, GPU 지원 | 엔터프라이즈 |
| **Qdrant** | 오픈소스 | Rust 기반, 고성능 | 실시간 검색 |

### 4.4 VectorDB 선택 가이드

```text
개발/학습/프로토타입
    └── ChromaDB (권장) - 설치 간편, 인메모리 가능

프로덕션/대규모 서비스
    └── Pinecone - 관리 부담 없음, 자동 확장

비용 절감/온프레미스
    └── Milvus, Qdrant - 자체 인프라 구축
```

---

## 5. ChromaDB 심화

### 5.1 ChromaDB 기본 사용법

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 인메모리 VectorDB 생성
vectorstore = Chroma.from_documents(
    documents=doc_chunks,
    embedding=embeddings,
    collection_name="my_collection"
)
```

### 5.2 영구 저장

```python
# 디스크에 저장
vectorstore = Chroma.from_documents(
    documents=doc_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 저장 경로
)

# 나중에 로드
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

### 5.3 검색 유형

**1. 기본 유사도 검색 (Similarity)**

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 상위 3개
)
results = retriever.invoke("질문")
```

**2. MMR 검색 (Maximal Marginal Relevance)**

다양성을 고려한 검색. 유사하면서도 서로 다른 내용의 문서를 반환합니다.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # 최종 반환 수
        "fetch_k": 10,    # 후보 문서 수
        "lambda_mult": 0.5  # 0=다양성, 1=관련성
    }
)
```

**3. 유사도 점수 포함 검색**

```python
results = vectorstore.similarity_search_with_score(
    query="질문",
    k=3
)
for doc, score in results:
    print(f"점수: {score:.4f} - {doc.page_content[:50]}...")
```

### 5.4 메타데이터 필터링

```python
# 특정 조건의 문서만 검색
results = vectorstore.similarity_search(
    query="RAG 설명",
    k=3,
    filter={"source": "rag_guide.pdf"}  # 특정 문서만
)

# 복합 필터
results = vectorstore.similarity_search(
    query="질문",
    filter={
        "$and": [
            {"topic": "AI"},
            {"year": {"$gte": 2023}}
        ]
    }
)
```

### 5.5 ChromaDB 아키텍처

```text
ChromaDB 구조:
┌─────────────────────────────────────────┐
│              Collection                  │
│  ┌─────────────────────────────────┐    │
│  │  Document 1                      │    │
│  │  - id: "doc1"                    │    │
│  │  - embedding: [0.23, 0.45, ...]  │    │
│  │  - content: "텍스트..."          │    │
│  │  - metadata: {source: "a.pdf"}   │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │  Document 2                      │    │
│  │  ...                             │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

---

## 6. 실무 적용 가이드

### 6.1 임베딩 캐싱

API 호출 비용을 줄이기 위한 캐싱 전략:

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# 캐시 저장소 설정
store = LocalFileStore("./embedding_cache")

# 캐시 적용 임베딩
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache=store,
    namespace="my_embeddings"
)
```

### 6.2 배치 처리

대량의 문서를 처리할 때:

```python
# 배치 크기 설정으로 API 호출 최적화
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size=1000  # 한 번에 처리할 텍스트 수
)
```

### 6.3 하이브리드 검색

키워드 검색과 벡터 검색을 결합:

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 (키워드) + Vector (의미) 결합
bm25_retriever = BM25Retriever.from_documents(documents)
vector_retriever = vectorstore.as_retriever()

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 벡터 검색에 가중치
)
```

### 6.4 성능 모니터링 지표

| 지표 | 설명 | 목표값 |
|------|------|--------|
| 검색 레이턴시 | 쿼리 응답 시간 | < 100ms |
| Recall@K | K개 중 관련 문서 비율 | > 0.8 |
| MRR | 첫 번째 관련 문서 순위 | > 0.7 |
| 저장 용량 | 벡터 저장 크기 | 모니터링 |

---

## 핵심 요약

###  체크리스트

- [ ] 프로젝트에 적합한 임베딩 모델 선택
- [ ] 한국어 프로젝트는 다국어 모델 고려 (BGE-M3)
- [ ] 개발 단계: ChromaDB 인메모리
- [ ] 프로덕션: 영구 저장 또는 클라우드 VectorDB
- [ ] 메타데이터 필터링 활용
- [ ] 캐싱으로 비용 최적화

###  포인트

1. **임베딩은 의미를 숫자로 변환** - 유사한 의미 = 가까운 벡터
2. **코사인 유사도가 표준** - 방향 기반 비교
3. **ChromaDB로 시작** - 간편하고 학습에 적합
4. **메타데이터가 핵심** - 필터링과 출처 추적에 필수


