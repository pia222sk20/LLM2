# LangGraph를 활용한 상태 기반 RAG 에이전트

## 목차

1. LangGraph 개요
2. 핵심 개념: State, Node, Edge
3. StateGraph 상세 설명
4. RAG vs LangGraph RAG
5. 적용 패턴

---

## 1. LangGraph 개요

### 1.1 LangGraph란?

LangGraph는 LangChain 위에 구축된 상태 기반 에이전트 프레임워크입니다.
복잡한 AI 워크플로우를 그래프 구조로 표현하고 실행할 수 있습니다.

기존 LangChain (Chain): 입력 -> A -> B -> C -> 출력 (선형적)
LangGraph (Graph): 분기, 순환이 가능한 그래프 구조

### 1.2 LangGraph가 필요한 이유

| 상황 | Chain의 한계 | LangGraph 해결책 |
|------|-------------|-----------------|
| 조건부 분기 | 복잡한 로직 필요 | 조건부 엣지 |
| 반복 실행 | 구현 어려움 | 순환(Cycle) 지원 |
| 상태 관리 | 수동 관리 필요 | 자동 상태 관리 |
| 에러 복구 | 직접 구현 | 내장 재시도 메커니즘 |

### 1.3 LangGraph의 핵심 철학

모든 AI 에이전트는 상태 머신(State Machine)이다.

상태(State)는 모든 노드가 공유하는 데이터이며,
각 노드는 상태를 읽고 업데이트합니다.

---

## 2. 핵심 개념: State, Node, Edge

### 2.1 State (상태)

State는 에이전트가 작업을 수행하면서 유지하는 데이터 구조입니다.

```python
from typing import TypedDict, List
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str              # 사용자 질문
    documents: List[Document]  # 검색된 문서
    context: str               # 추출된 컨텍스트
    answer: str                # 생성된 답변
```

State의 특징:
- 모든 노드가 공유하는 데이터
- 각 노드는 State를 읽고 업데이트
- TypedDict로 타입 안전성 확보

### 2.2 Node (노드)

Node는 실제 작업을 수행하는 함수입니다.

```python
def retrieve_node(state: RAGState) -> dict:
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

def generate_node(state: RAGState) -> dict:
    context = format_docs(state["documents"])
    answer = llm.invoke(context + state["question"])
    return {"answer": answer}
```

노드의 규칙:
1. State를 입력으로 받음
2. 딕셔너리를 반환 (상태 업데이트)
3. 반환된 딕셔너리가 기존 State와 병합됨

### 2.3 Edge (엣지)

Edge는 노드 간의 연결(흐름)을 정의합니다.

```python
# 일반 엣지: A -> B
graph.add_edge("retrieve", "generate")

# 조건부 엣지: 조건에 따라 다른 노드로 분기
graph.add_conditional_edges(
    "grade",
    decide_next_step,
    {"relevant": "generate", "not_relevant": "search"}
)
```

---

## 3. StateGraph 상세 설명

### 3.1 StateGraph 생성

```python
from langgraph.graph import StateGraph, START, END

# 1. 그래프 생성
graph = StateGraph(RAGState)

# 2. 노드 추가
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

# 3. 엣지 추가
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# 4. 컴파일
app = graph.compile()

# 5. 실행
result = app.invoke({"question": "RAG란?"})
```

### 3.2 그래프 구조

START -> RETRIEVE -> GENERATE -> END

- RETRIEVE 노드: state["question"]을 읽고 state["documents"]를 업데이트
- GENERATE 노드: state["documents"]를 읽고 state["answer"]를 업데이트

### 3.3 조건부 엣지

동적으로 다음 노드를 결정합니다.

```python
def decide_to_generate(state: RAGState) -> str:
    if state["documents"]:
        return "generate"
    else:
        return "web_search"

graph.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"generate": "generate", "web_search": "web_search"}
)
```

### 3.4 실행 방법

```python
# 단일 실행
result = app.invoke({"question": "RAG란?"})

# 스트리밍 실행
for output in app.stream({"question": "RAG란?"}):
    for node_name, node_output in output.items():
        print(f"노드 {node_name} 완료: {node_output}")
```

---

## 4. 기초 RAG vs LangGraph RAG

### 4.1 기초 RAG (Chain 방식)

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
answer = rag_chain.invoke("RAG란?")
```

특징: 선형적 실행, 간단하고 직관적, 조건부 분기 어려움

### 4.2 LangGraph RAG (Graph 방식)

```python
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
app = graph.compile()
result = app.invoke({"question": "RAG란?"})
```

특징: 상태 기반 실행, 조건부 분기 가능, 순환 지원, 디버깅 용이

### 4.3 비교

| 항목 | 기초 RAG | LangGraph RAG |
|------|----------|---------------|
| 구조 | 선형 | 그래프 |
| 상태 관리 | 수동 | 자동 |
| 조건부 분기 | 복잡 | 내장 지원 |
| 디버깅 | 어려움 | 노드별 추적 가능 |

---

## 5. 적용 패턴

### 5.1 패턴 1: 기본 RAG

START -> retrieve -> generate -> END

가장 단순한 패턴입니다.

### 5.2 패턴 2: 문서 등급 평가 RAG

START -> retrieve -> grade_documents -> generate -> END

검색된 문서의 관련성을 평가하고, 관련 없으면 웹 검색으로 전환합니다.

### 5.3 패턴 3: 답변 검증 RAG (Self-RAG)

START -> retrieve -> generate -> check_hallucination -> END

생성된 답변의 환각을 검사하고, 문제가 있으면 재생성합니다.

---

## 요약

체크리스트:
- TypedDict로 State 스키마 정의
- 노드 함수는 State를 받아 dict를 반환
- add_edge()로 노드 연결
- add_conditional_edges()로 조건부 분기
- compile() 후 invoke() 또는 stream()으로 실행

포인트:
1. State가 핵심 - 모든 노드가 공유하는 데이터 구조
2. 노드는 함수 - 상태를 읽고, 업데이트할 부분만 반환
3. 엣지는 흐름 - 노드 간 연결, 조건부 분기 가능
4. 컴파일 필수 - graph.compile()로 실행 가능한 앱 생성

---

## NEXT
- CRAG (Corrective RAG) 패턴
- 문서 관련성 평가
- 웹 검색 폴백 구현
