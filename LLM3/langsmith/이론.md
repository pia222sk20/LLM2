# LangSmith 모니터링

## 목차

1. [LLM 모니터링의 필요성](#1-llm-모니터링의-필요성)
2. [LangSmith 개요](#2-langsmith-개요)
3. [Tracing (추적)](#3-tracing-추적)
4. [평가 (Evaluation)](#4-평가-evaluation)
5. [프롬프트 관리](#5-프롬프트-관리)
6. [적용 가이드](#6-적용-가이드)

---

## 1. LLM 모니터링의 필요성

### 1.1 LLM 애플리케이션의 특수성

기존 소프트웨어와 LLM 애플리케이션의 차이:

```text
기존 소프트웨어:
    입력 A → 함수 f() → 항상 출력 B (결정적)

LLM 애플리케이션:
    입력 A → LLM → 출력 B, C, D... (비결정적)
                    ↑
              매번 다를 수 있음
```

### 1.2 모니터링이 필요한 이유

| 문제 | 모니터링 없이 | 모니터링 있으면 |
|------|--------------|----------------|
| **환각 발생** | 사용자 불만 후 인지 | 즉시 감지 |
| **응답 지연** | 원인 파악 어려움 | 병목 지점 확인 |
| **비용 초과** | 청구서에서 발견 | 실시간 추적 |
| **품질 저하** | 점진적 악화 | 추이 모니터링 |

### 1.3 LLM 모니터링 핵심 지표

```text
┌─────────────────────────────────────────────────────┐
│                  핵심 모니터링 지표                   │
├─────────────────────────────────────────────────────┤
│  성능 (Performance)                               │
│    - Latency (응답 시간)                            │
│    - Throughput (처리량)                            │
│    - Error Rate (오류율)                            │
├─────────────────────────────────────────────────────┤
│  비용 (Cost)                                      │
│    - Token Usage (토큰 사용량)                       │
│    - API Calls (호출 횟수)                          │
│    - Cost per Query (쿼리당 비용)                    │
├─────────────────────────────────────────────────────┤
│  품질 (Quality)                                   │
│    - Relevance (관련성)                             │
│    - Accuracy (정확도)                              │
│    - User Feedback (사용자 피드백)                   │
└─────────────────────────────────────────────────────┘
```

---

## 2. LangSmith 개요

### 2.1 LangSmith란?

**LangSmith**는 LangChain에서 제공하는 LLM 애플리케이션 개발 플랫폼입니다.

```text
LangSmith 기능:
├── Tracing: 모든 LLM 호출 추적
├── Evaluation: 자동화된 품질 평가
├── Datasets: 테스트 데이터 관리
├── Hub: 프롬프트 공유 및 버전 관리
└── Monitoring: 프로덕션 모니터링
```

### 2.2 LangSmith API 키 발급 방법 

#### Step 1: LangSmith 회원가입

1. **LangSmith 웹사이트 접속**
   - 브라우저에서 [https://smith.langchain.com](https://smith.langchain.com) 접속

2. **계정 생성**
   - "Sign Up" 버튼 클릭
   - GitHub, Google, 또는 이메일로 회원가입
   - 이메일 인증 완료 (이메일 가입 시)

```text
┌─────────────────────────────────────────────────────┐
│          LangSmith 로그인 화면                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│          Continue with GitHub                     │
│          Continue with Google                     │
│           Continue with Email                     │
│                                                     │
│         ─────────── OR ───────────                  │
│                                                     │
│         Already have an account? Sign In            │
└─────────────────────────────────────────────────────┘
```

#### Step 2: API 키 생성

1. **Settings 메뉴 접근**
   - 로그인 후 좌측 하단 "Settings" 클릭
   - 또는 프로필 아이콘 클릭 → "Settings"

2. **API Keys 섹션**
   - "API Keys" 탭 선택
   - "Create API Key" 버튼 클릭

3. **키 생성 및 복사**
   - 키 이름 입력 (예: "my-rag-project")
   - "Create" 클릭
   - ** 중요: 생성된 키는 한 번만 표시됩니다. 반드시 복사하여 안전하게 저장하세요!**

```text
┌─────────────────────────────────────────────────────┐
│  Settings > API Keys                                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Your API Keys                                      │
│  ┌───────────────────────────────────────────────┐ │
│  │ Name: my-rag-project                          │ │
│  │ Key:  lsv2_pt_a1b2c3d4e5f6g7h8...  [Copy]    │ │
│  │ Created: 2024-01-15                           │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  [+ Create API Key]                                 │
└─────────────────────────────────────────────────────┘
```

#### Step 3: 프로젝트 생성 (선택사항)

1. **Projects 메뉴**
   - 좌측 메뉴에서 "Projects" 클릭
   - "+ New Project" 버튼 클릭

2. **프로젝트 정보 입력**
   - 프로젝트 이름 입력
   - 설명 추가 (선택)
   - "Create" 클릭

### 2.3 환경 변수 설정

#### Python 코드에서 직접 설정

```python
import os

# 환경 변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_..."  # 발급받은 API 키
os.environ["LANGCHAIN_PROJECT"] = "my-project"   # 프로젝트명
```

#### .env 파일 사용 (권장)

프로젝트 루트에 `.env` 파일 생성:

```env
# LangSmith 설정
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=llm-rag-project
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

#### 환경 변수 설명

| 환경 변수 | 필수 | 설명 |
|----------|------|------|
| `LANGCHAIN_TRACING_V2` |  | `true`로 설정하면 추적 활성화 |
| `LANGCHAIN_API_KEY` |  | LangSmith API 키 |
| `LANGCHAIN_PROJECT` |  | 프로젝트 이름 (미지정시 "default") |
| `LANGCHAIN_ENDPOINT` | | API 엔드포인트 (기본값 사용 권장) |

### 2.4 요금제 안내

| 플랜 | 가격 | Trace 수 | 주요 기능 |
|------|------|----------|----------|
| **Free** | $0/월 | 5,000/월 | 기본 추적, 평가 |
| **Developer** | $39/월 | 50,000/월 | 팀 협업, 우선 지원 |
| **Plus** | 사용량 기반 | 무제한 | 엔터프라이즈 기능 |

>  **팁**: 학습 및 개발 단계에서는 Free 플랜으로 충분합니다!

### 2.5 연결 확인

```python
from langsmith import Client

# 연결 테스트
client = Client()
print(" LangSmith 연결 성공!")
print(f"현재 프로젝트: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
```

### 2.6 LangSmith 대시보드

```text
LangSmith Dashboard:
┌─────────────────────────────────────────────────────┐
│ Projects                                            │
│ ├── llm-rag-project                                │
│ │   ├── Runs: 1,234 (최근 24시간)                   │
│ │   ├── Avg Latency: 2.3s                          │
│ │   ├── Error Rate: 0.5%                           │
│ │   └── Total Cost: $12.50                         │
│ └── other-project                                  │
├─────────────────────────────────────────────────────┤
│ Recent Runs                                         │
│ ├── run_abc123 | 2.1s | $0.002 |                 │
│ ├── run_def456 | 3.5s | $0.003 |                 │
│ └── run_ghi789 | 1.2s | $0.001 |  Error          │
└─────────────────────────────────────────────────────┘
```

### 2.7 트러블슈팅

#### 자주 발생하는 오류

| 오류 메시지 | 원인 | 해결 방법 |
|------------|------|----------|
| `Invalid API key` | API 키가 잘못됨 | 키를 다시 확인/재발급 |
| `Project not found` | 프로젝트명 오류 | 프로젝트명 확인 |
| `Rate limit exceeded` | 호출 한도 초과 | 잠시 후 재시도 또는 플랜 업그레이드 |
| `Connection refused` | 네트워크 문제 | 방화벽/프록시 확인 |

#### API 키 보안 주의사항

```text
API 키 보안 체크리스트:

 .env 파일을 .gitignore에 추가
 API 키를 코드에 직접 하드코딩하지 않기
 팀원 간 키 공유 시 안전한 방법 사용
 주기적으로 키 로테이션
 불필요한 키는 즉시 삭제
```

**.gitignore 예시:**
```
.env
.env.local
*.key
```

### 2.3 LangSmith 대시보드

```text
LangSmith Dashboard:
┌─────────────────────────────────────────────────────┐
│ Projects                                            │
│ ├── llm-rag-project                                │
│ │   ├── Runs: 1,234 (최근 24시간)                   │
│ │   ├── Avg Latency: 2.3s                          │
│ │   ├── Error Rate: 0.5%                           │
│ │   └── Total Cost: $12.50                         │
│ └── other-project                                  │
├─────────────────────────────────────────────────────┤
│ Recent Runs                                         │
│ ├── run_abc123 | 2.1s | $0.002 |                 │
│ ├── run_def456 | 3.5s | $0.003 |                 │
│ └── run_ghi789 | 1.2s | $0.001 |  Error          │
└─────────────────────────────────────────────────────┘
```

---

## 3. Tracing (추적)

### 3.1 자동 추적

LangChain 컴포넌트는 자동으로 추적됩니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# LANGCHAIN_TRACING_V2=true면 자동 추적
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("질문: {question}")

chain = prompt | llm
response = chain.invoke({"question": "RAG란?"})
# → LangSmith에 자동으로 기록됨
```

### 3.2 추적 정보 구조

```text
Run (실행):
├── id: "run_abc123"
├── name: "RunnableSequence"
├── inputs: {"question": "RAG란?"}
├── outputs: {"content": "RAG는..."}
├── start_time: "2024-01-01T10:00:00"
├── end_time: "2024-01-01T10:00:02"
├── latency: 2.0s
├── tokens: {"input": 50, "output": 100}
├── cost: $0.002
└── child_runs: [
    ├── ChatPromptTemplate (0.01s)
    └── ChatOpenAI (1.99s)
]
```

### 3.3 커스텀 추적

```python
from langsmith.run_helpers import traceable

@traceable(name="custom_rag_pipeline")
def my_rag_function(question: str) -> str:
    """커스텀 함수도 추적 가능"""
    docs = retriever.invoke(question)
    answer = llm.invoke(format_prompt(docs, question))
    return answer

# 호출 시 자동으로 LangSmith에 기록
result = my_rag_function("RAG란?")
```

### 3.4 메타데이터 추가

```python
from langchain_core.runnables import RunnableConfig

# 추가 메타데이터와 함께 실행
config = RunnableConfig(
    metadata={
        "user_id": "user_123",
        "session_id": "sess_456",
        "version": "v1.2.0"
    },
    tags=["production", "rag"]
)

response = chain.invoke({"question": "질문"}, config=config)
```

---

## 4. 평가 (Evaluation)

### 4.1 평가의 중요성

```text
"측정하지 않으면 개선할 수 없다"

RAG 시스템 평가 지표:
├── Retrieval Quality: 검색된 문서의 관련성
├── Answer Relevance: 답변이 질문과 관련 있는가
├── Answer Correctness: 답변이 정확한가
├── Faithfulness: 답변이 컨텍스트에 기반하는가
└── Latency: 응답 시간
```

### 4.2 데이터셋 생성

```python
from langsmith import Client

client = Client()

# 평가용 데이터셋 생성
dataset = client.create_dataset(
    dataset_name="rag_evaluation_set",
    description="RAG 시스템 평가용 데이터셋"
)

# 예제 추가
examples = [
    {
        "inputs": {"question": "RAG란 무엇인가요?"},
        "outputs": {"answer": "RAG는 검색 증강 생성 기술입니다."}
    },
    {
        "inputs": {"question": "LangChain의 주요 기능은?"},
        "outputs": {"answer": "LangChain은 LLM 애플리케이션 프레임워크입니다."}
    }
]

for ex in examples:
    client.create_example(
        inputs=ex["inputs"],
        outputs=ex["outputs"],
        dataset_id=dataset.id
    )
```

### 4.3 평가 함수 정의

```python
from langsmith.evaluation import evaluate

def answer_relevance_evaluator(run, example):
    """답변 관련성 평가"""
    # LLM으로 평가
    evaluation_prompt = f"""
    질문: {example.inputs['question']}
    답변: {run.outputs['answer']}
    
    답변이 질문과 관련이 있으면 1, 없으면 0을 반환하세요.
    """
    score = llm.invoke(evaluation_prompt)
    return {"score": int(score), "key": "relevance"}

def latency_evaluator(run, example):
    """응답 시간 평가"""
    latency = run.end_time - run.start_time
    return {
        "score": 1 if latency < 3.0 else 0,
        "key": "latency_ok"
    }
```

### 4.4 평가 실행

```python
# 평가 실행
results = evaluate(
    my_rag_function,                    # 평가할 함수
    data="rag_evaluation_set",          # 데이터셋
    evaluators=[
        answer_relevance_evaluator,
        latency_evaluator
    ],
    experiment_prefix="rag_v1",         # 실험 이름
    max_concurrency=4                   # 병렬 실행
)

print(f"평균 관련성: {results.aggregate_metrics['relevance']}")
print(f"평균 응답시간: {results.aggregate_metrics['latency_ok']}")
```

---

## 5. 프롬프트 관리

### 5.1 LangSmith Hub

프롬프트를 중앙에서 관리하고 버전 관리합니다.

```python
from langchain import hub

# 프롬프트 가져오기
prompt = hub.pull("rlm/rag-prompt")

# 프롬프트 저장
hub.push("my-org/my-rag-prompt", my_prompt)
```

### 5.2 프롬프트 버전 관리

```text
my-rag-prompt:
├── v1 (2024-01-01): 기본 버전
├── v2 (2024-01-15): 규칙 추가
├── v3 (2024-02-01): 출력 형식 개선 ← latest
└── v4 (draft): 테스트 중
```

### 5.3 A/B 테스트

```python
# 두 프롬프트 비교
prompt_a = hub.pull("my-org/rag-prompt:v2")
prompt_b = hub.pull("my-org/rag-prompt:v3")

# 각각으로 평가 실행
results_a = evaluate(chain_with_prompt_a, data="test_set", experiment_prefix="prompt_v2")
results_b = evaluate(chain_with_prompt_b, data="test_set", experiment_prefix="prompt_v3")

# 결과 비교
print(f"v2 정확도: {results_a.aggregate_metrics['accuracy']}")
print(f"v3 정확도: {results_b.aggregate_metrics['accuracy']}")
```

---

## 6. 적용 가이드

### 6.1 모니터링 대시보드 설계

```text
프로덕션 모니터링 대시보드:

┌─────────────────────────────────────────────────────┐
│ 실시간 지표 (최근 1시간)                          │
├─────────────────────────────────────────────────────┤
│ 요청 수: 1,234    │ 평균 응답시간: 2.3s              │
│ 성공률: 99.5%     │ 평균 비용: $0.002/요청           │
├─────────────────────────────────────────────────────┤
│  응답시간 추이                                    │
│ ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█                             │
├─────────────────────────────────────────────────────┤
│  알림                                             │
│ - 10:30 응답시간 5초 초과 (3건)                      │
│ - 10:45 토큰 한도 90% 도달                          │
└─────────────────────────────────────────────────────┘
```

### 6.2 알림 설정

```python
# 커스텀 알림 로직
def check_and_alert(run):
    """실행 결과 체크 및 알림"""
    
    # 응답 시간 체크
    if run.latency > 5.0:
        send_alert(f"느린 응답: {run.id} - {run.latency}s")
    
    # 에러 체크
    if run.error:
        send_alert(f"에러 발생: {run.id} - {run.error}")
    
    # 비용 체크
    if run.total_cost > 0.01:
        send_alert(f"높은 비용: {run.id} - ${run.total_cost}")
```

### 6.3 비용 최적화

```python
from langsmith import Client

client = Client()

# 비용 분석
runs = client.list_runs(
    project_name="my-project",
    start_time=datetime.now() - timedelta(days=7)
)

total_cost = sum(run.total_cost or 0 for run in runs)
avg_cost = total_cost / len(list(runs))

print(f"주간 총 비용: ${total_cost:.2f}")
print(f"요청당 평균 비용: ${avg_cost:.4f}")

# 비용이 높은 요청 분석
expensive_runs = [r for r in runs if (r.total_cost or 0) > 0.01]
for run in expensive_runs[:5]:
    print(f"- {run.id}: ${run.total_cost:.4f} ({run.total_tokens} tokens)")
```

### 6.4 디버깅 워크플로우

```text
문제 발생 시 디버깅 순서:

1. LangSmith에서 해당 Run 찾기
   └── 필터: 시간, 에러, 사용자 ID

2. Run 상세 정보 확인
   ├── 입력값 확인
   ├── 각 단계별 출력 확인
   └── 에러 메시지 확인

3. 재현
   └── 같은 입력으로 로컬에서 테스트

4. 수정 및 검증
   └── 평가 데이터셋에 케이스 추가
```

### 6.5 체크리스트

**개발 단계:**

- [ ] LANGCHAIN_TRACING_V2=true 설정
- [ ] 프로젝트 이름 설정
- [ ] 주요 함수에 @traceable 적용

**테스트 단계:**

- [ ] 평가 데이터셋 생성
- [ ] 평가 함수 정의
- [ ] 베이스라인 측정

**프로덕션 단계:**

- [ ] 알림 설정
- [ ] 비용 한도 설정
- [ ] 대시보드 구성

---

## 핵심 요약

### 체크리스트

- [ ] LangSmith API 키 발급
- [ ] 환경 변수 설정
- [ ] 프로젝트 생성
- [ ] 평가 데이터셋 구축
- [ ] 모니터링 알림 설정

### 포인트

1. **추적은 필수**: 모든 LLM 호출 기록
2. **평가 자동화**: 데이터셋 기반 품질 관리
3. **비용 모니터링**: 예상치 못한 비용 방지
4. **프롬프트 버전 관리**: A/B 테스트로 개선



