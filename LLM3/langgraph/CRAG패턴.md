# Corrective RAG (CRAG) 패턴 이론

## 목차

1. [CRAG 개요](#1-crag-개요)
2. [CRAG 아키텍처](#2-crag-아키텍처)
3. [문서 관련성 평가 (Grading)](#3-문서-관련성-평가-grading)
4. [조건부 엣지 상세](#4-조건부-엣지-상세)
5. [웹 검색 폴백](#5-웹-검색-폴백)

---

## 1. CRAG 개요

### 1.1 CRAG란?

**Corrective RAG (CRAG)**는 검색된 문서의 품질을 자체적으로 평가하고, 필요시 **자기 교정(Self-Correction)**을 수행하는 고급 RAG 패턴입니다.

```text
기존 RAG:
    검색 → 생성 (검색 품질 무관)

CRAG:
    검색 → 평가 → [충분하면 생성 | 부족하면 웹 검색 → 생성]
```

### 1.2 CRAG가 해결하는 문제

| 문제 | 기존 RAG | CRAG 해결책 |
|------|----------|-------------|
| 관련 없는 문서 검색 | 그대로 사용 → 환각 | LLM으로 관련성 평가 |
| 내부 문서 부족 | 답변 불가 | 웹 검색 폴백 |
| 답변 품질 불균일 | 관리 어려움 | 자동 품질 게이트 |

### 1.3 CRAG의 핵심 아이디어

```text
"검색 결과를 맹목적으로 신뢰하지 않는다"

1. 검색된 문서가 질문과 관련이 있는가? → LLM이 평가
2. 관련 있는 문서가 충분한가? → 개수/품질 체크
3. 부족하다면? → 외부 소스(웹)에서 보충
```

---

## 2. CRAG 아키텍처

### 2.1 전체 흐름도

```text
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   RETRIEVE   │  ← 내부 VectorDB 검색
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    GRADE     │  ← LLM으로 문서 관련성 평가
                    │  DOCUMENTS   │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         관련 있음                    관련 없음
              │                         │
              ▼                         ▼
      ┌──────────────┐         ┌──────────────┐
      │   GENERATE   │         │  WEB_SEARCH  │  ← 외부 웹 검색
      └──────┬───────┘         └──────┬───────┘
              │                       │
              │                       ▼
              │               ┌──────────────┐
              │               │   GENERATE   │
              │               └──────┬───────┘
              │                      │
              └──────────────────────┴─────────┐
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │     END      │
                                        └──────────────┘
```

### 2.2 CRAG State 설계

```python
from typing import TypedDict, List, Literal

class CRAGState(TypedDict):
    """CRAG 에이전트 상태"""
    question: str                      # 사용자 질문
    documents: List[Document]          # 검색된 문서
    filtered_documents: List[Document] # 필터링된 관련 문서
    web_results: List[Document]        # 웹 검색 결과
    generation_source: Literal["internal", "web", "hybrid"]
    answer: str                        # 최종 답변
```

### 2.3 노드별 역할

| 노드 | 역할 | 입력 | 출력 |
|------|------|------|------|
| `retrieve` | 내부 문서 검색 | question | documents |
| `grade_documents` | 관련성 평가 | documents | filtered_documents |
| `web_search` | 웹 검색 | question | web_results |
| `generate` | 답변 생성 | filtered/web docs | answer |

---

## 3. 문서 관련성 평가 (Grading)

### 3.1 LLM 기반 문서 평가

LLM을 사용하여 각 문서가 질문과 관련이 있는지 평가합니다.

```python
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    """문서 관련성 평가 스키마"""
    is_relevant: Literal["yes", "no"] = Field(
        description="문서가 질문과 관련이 있으면 'yes', 없으면 'no'"
    )

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 문서 관련성 평가 전문가입니다.
주어진 문서가 질문에 답변하는 데 도움이 되는지 평가하세요.
관련이 있으면 'yes', 없으면 'no'로만 답하세요."""),
    ("human", """문서: {document}

질문: {question}

관련성:""")
])

# 구조화된 출력으로 LLM 호출
grader = grade_prompt | llm.with_structured_output(GradeDocuments)
```

### 3.2 평가 기준

```text
관련성 평가 기준:

 관련 있음 (yes):
   - 문서가 질문의 핵심 주제를 다룸
   - 질문에 직접적인 답변 정보 포함
   - 질문 이해에 도움이 되는 배경 정보

 관련 없음 (no):
   - 질문과 완전히 다른 주제
   - 키워드만 일치하고 맥락이 다름
   - 답변에 도움이 되지 않는 정보
```

### 3.3 평가 노드 구현

```python
def grade_documents_node(state: CRAGState) -> dict:
    """검색된 문서의 관련성을 평가하고 필터링"""
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for doc in documents:
        result = grader.invoke({
            "document": doc.page_content,
            "question": question
        })
        if result.is_relevant == "yes":
            filtered_docs.append(doc)
    
    return {"filtered_documents": filtered_docs}
```

---

## 4. 조건부 엣지 상세

### 4.1 조건부 분기의 원리

조건부 엣지는 **조건 함수**의 반환값에 따라 다음 노드를 결정합니다.

```python
def decide_to_generate(state: CRAGState) -> Literal["generate", "web_search"]:
    """다음 단계 결정 함수"""
    filtered_docs = state["filtered_documents"]
    
    # 관련 문서가 충분한가?
    if len(filtered_docs) >= 1:
        return "generate"    # 내부 문서로 생성
    else:
        return "web_search"  # 웹 검색 필요
```

### 4.2 그래프에 조건부 엣지 추가

```python
graph.add_conditional_edges(
    "grade_documents",      # 시작 노드
    decide_to_generate,     # 조건 함수
    {
        "generate": "generate",      # 함수 반환값 → 목적지 노드
        "web_search": "web_search"
    }
)
```

### 4.3 조건부 엣지 시각화

```text
         grade_documents
              │
              ▼
        ┌─────────────┐
        │ 조건 함수    │
        │ 실행        │
        └─────┬───────┘
              │
    ┌─────────┴─────────┐
    │                   │
"generate"         "web_search"
    │                   │
    ▼                   ▼
generate            web_search
```

### 4.4 다중 조건 처리

```python
def complex_decision(state: CRAGState) -> str:
    """복잡한 다중 조건 처리"""
    docs = state["filtered_documents"]
    confidence = calculate_confidence(docs)
    
    if len(docs) >= 3 and confidence > 0.8:
        return "generate_confident"
    elif len(docs) >= 1:
        return "generate_with_disclaimer"
    elif state.get("web_search_done"):
        return "fallback_response"
    else:
        return "web_search"
```

---

## 5. 웹 검색 폴백

### 5.1 폴백(Fallback)이란?

**폴백**은 주요 방법이 실패했을 때 대체 방법을 사용하는 패턴입니다.

```text
CRAG의 폴백 전략:

1차: 내부 VectorDB 검색
        │
        ▼
    ┌───────────┐
    │ 충분한가? │
    └─────┬─────┘
          │
    ┌─────┴─────┐
    예          아니오
    │           │
    ▼           ▼
  생성     2차: 웹 검색 (폴백)
                │
                ▼
              생성
```

### 5.2 웹 검색 통합

```python
def web_search_node(state: CRAGState) -> dict:
    """웹 검색으로 추가 정보 획득"""
    question = state["question"]
    
    # Tavily, Google 등 웹 검색 API 활용
    web_results = tavily_search.invoke(question)
    
    # Document 형태로 변환
    web_docs = [
        Document(
            page_content=result["content"],
            metadata={"source": result["url"], "type": "web"}
        )
        for result in web_results
    ]
    
    return {
        "web_results": web_docs,
        "generation_source": "web"
    }
```

### 5.3 하이브리드 생성

내부 문서와 웹 검색 결과를 모두 활용합니다.

```python
def generate_node(state: CRAGState) -> dict:
    """내부 문서 + 웹 결과로 답변 생성"""
    
    # 소스에 따라 문서 선택
    if state["generation_source"] == "internal":
        docs = state["filtered_documents"]
    elif state["generation_source"] == "web":
        docs = state["web_results"]
    else:  # hybrid
        docs = state["filtered_documents"] + state["web_results"]
    
    context = format_docs(docs)
    answer = llm.invoke(prompt.format(context=context, question=state["question"]))
    
    return {"answer": answer}
```

---

## 6.적용 가이드

### 6.1 CRAG 구현 체크리스트

1. **State 설계**
   - [ ] 필요한 모든 필드 정의
   - [ ] 타입 힌트 명시

2. **평가 프롬프트**
   - [ ] 명확한 평가 기준 설정
   - [ ] 구조화된 출력 스키마 정의

3. **조건부 로직**
   - [ ] 분기 조건 명확히 정의
   - [ ] 모든 경로에 대한 처리

4. **폴백 전략**
   - [ ] 웹 검색 API 연동
   - [ ] 결과 통합 로직

### 6.2 성능 최적화

```python
# 1. 병렬 평가로 속도 향상
import asyncio

async def grade_documents_parallel(documents, question):
    tasks = [grade_single_doc(doc, question) for doc in documents]
    results = await asyncio.gather(*tasks)
    return [doc for doc, is_relevant in zip(documents, results) if is_relevant]

# 2. 캐싱으로 중복 평가 방지
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_grade(doc_hash, question_hash):
    return grader.invoke(...)

# 3. 배치 평가
def batch_grade(documents, question, batch_size=5):
    # 여러 문서를 한 번에 평가
    pass
```

### 6.3 모니터링 포인트

| 메트릭 | 설명 | 목표값 |
|--------|------|--------|
| 내부 문서 사용률 | 웹 검색 없이 답변한 비율 | > 70% |
| 평균 관련 문서 수 | 필터링 후 남은 문서 | 1-3개 |
| 웹 검색 레이턴시 | 폴백 시 추가 시간 | < 2초 |
| 답변 품질 점수 | 사용자 평가 기반 | > 4/5 |

### 6.4 확장 패턴

```text
기본 CRAG:
    Retrieve → Grade → [Generate | WebSearch → Generate]

확장 CRAG (품질 검증 추가):
    Retrieve → Grade → [Generate | WebSearch → Generate] → Verify → [End | Regenerate]
                                                              ↑           │
                                                              └───────────┘
```