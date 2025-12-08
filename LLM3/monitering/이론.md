# 로컬 모니터링 시스템 (API 키 불필요)

## 목차

1. [개요](#1-개요)
2. [로컬 모니터링의 필요성](#2-로컬-모니터링의-필요성)
3. [SQLite 기반 추적 시스템](#3-sqlite-기반-추적-시스템)
4. [커스텀 콜백 핸들러](#4-커스텀-콜백-핸들러)
5. [메트릭 수집 및 분석](#5-메트릭-수집-및-분석)
6. [LangSmith vs 로컬 모니터링](#6-langsmith-vs-로컬-모니터링)
7. [적용 가이드](#7-적용-가이드)

---

## 1. 개요

### 1.1 이 코드의 목적

`06_local_monitoring.py`는 **외부 API 키 없이** LLM 애플리케이션을 모니터링하는 방법을 제공합니다.

```text
┌─────────────────────────────────────────────────────┐
│           API 키 요구사항 비교                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  langsmith_monitoring.py (원본)                  │
│  ├── OPENAI_API_KEY       필요                   │
│  └── LANGCHAIN_API_KEY    필요 (LangSmith)       │
│                                                     │
│  local_monitoring.py (대안)                      │
│  ├── OPENAI_API_KEY       필요                   │
│  └── 추가 API 키           불필요                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 1.2 핵심 기술 스택

| 기술 | 역할 | 장점 |
|------|------|------|
| **SQLite** | 데이터 저장 | 파일 하나로 관리, 무료 |
| **콜백 핸들러** | 이벤트 캡처 | 완전한 제어 |
| **Python 로깅** | 실시간 출력 | 간단, 즉각적 |

---

## 2. 로컬 모니터링의 필요성

### 2.1 LangSmith를 사용할 수 없는 상황

```text
로컬 모니터링이 필요한 경우:

1.  비용 문제
   └── 무료 티어 한도 초과 / 예산 제약

2.  보안/규정 요구사항
   └── 데이터를 외부 서버에 전송할 수 없음

3.  네트워크 제약
   └── 오프라인 환경 / 폐쇄망 환경

4.  목적
   └── 모니터링 원리를 직접 이해하고 싶음

5.  커스터마이징 필요
   └── 특수한 메트릭이나 분석 필요
```

### 2.2 로컬 모니터링의 장점

| 장점 | 설명 |
|------|------|
| **무료** | API 비용 없음, 제한 없음 |
| **프라이버시** | 모든 데이터가 로컬에 저장 |
| **오프라인** | 인터넷 연결 불필요 |
| **완전 제어** | 원하는 대로 커스터마이징 |
| **학습 가치** | 모니터링 시스템 이해 |

---

## 3. SQLite 기반 추적 시스템

### 3.1 SQLite를 선택한 이유

```text
왜 SQLite인가?

 설치 불필요 (Python 내장)
 파일 하나로 모든 데이터 관리
 SQL 쿼리로 복잡한 분석 가능
 백업 및 이동 용이 (파일 복사)
 경량 (수백MB 데이터도 빠르게 처리)
```

### 3.2 데이터베이스 스키마

```sql
-- 실행 추적 테이블
CREATE TABLE runs (
    id TEXT PRIMARY KEY,           -- UUID
    name TEXT,                     -- 실행 이름
    run_type TEXT,                 -- 'llm', 'chain', 'retriever'
    start_time TEXT,               -- ISO 형식 시작 시간
    end_time TEXT,                 -- ISO 형식 종료 시간
    duration_seconds REAL,         -- 소요 시간
    input_data TEXT,               -- JSON 입력
    output_data TEXT,              -- JSON 출력
    metadata TEXT,                 -- JSON 메타데이터
    status TEXT,                   -- 'running', 'success', 'error'
    error TEXT                     -- 에러 메시지
);

-- 메트릭 테이블
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY,
    run_id TEXT,                   -- runs.id 참조
    metric_name TEXT,              -- 메트릭 이름
    metric_value REAL,             -- 메트릭 값
    recorded_at TEXT               -- 기록 시간
);

-- 토큰 사용량 테이블
CREATE TABLE token_usage (
    id INTEGER PRIMARY KEY,
    run_id TEXT,                   -- runs.id 참조
    prompt_tokens INTEGER,         -- 입력 토큰
    completion_tokens INTEGER,     -- 출력 토큰
    total_tokens INTEGER,          -- 총 토큰
    estimated_cost REAL,           -- 추정 비용
    model TEXT,                    -- 모델명
    recorded_at TEXT               -- 기록 시간
);
```

### 3.3 추적 흐름

```text
LLM 호출 시작
      │
      ▼
┌─────────────────┐
│  start_run()    │ ← runs 테이블에 새 레코드 INSERT
│  - UUID 생성    │
│  - 시작 시간    │
│  - 입력 데이터  │
└────────┬────────┘
         │
         ▼
    [LLM 처리 중...]
         │
         ▼
┌─────────────────┐
│   end_run()     │ ← runs 테이블 UPDATE
│  - 종료 시간    │
│  - 소요 시간    │
│  - 출력 데이터  │
│  - 상태         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ record_token_   │ ← token_usage 테이블 INSERT
│   usage()       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ record_metric() │ ← metrics 테이블 INSERT
└─────────────────┘
```

---

## 4. 커스텀 콜백 핸들러

### 4.1 LangChain 콜백 시스템 이해

```text
LangChain 콜백 라이프사이클:

LLM 호출:
  on_llm_start() → [LLM 처리] → on_llm_end() 또는 on_llm_error()

체인 실행:
  on_chain_start() → [체인 처리] → on_chain_end()

검색:
  on_retriever_start() → [검색] → on_retriever_end()
```

### 4.2 콜백 핸들러 구현

```python
from langchain_core.callbacks import BaseCallbackHandler

class LocalMonitoringHandler(BaseCallbackHandler):
    """
    로컬 모니터링을 위한 콜백 핸들러
    """
    
    def __init__(self, trace_db, log_to_console=True):
        self.trace_db = trace_db
        self.log_to_console = log_to_console
        self.current_run_id = None
        self.start_time = None
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 호출 시작 시 실행"""
        # 1. DB에 실행 시작 기록
        self.current_run_id = self.trace_db.start_run(...)
        
        # 2. 콘솔에 로깅 (선택)
        if self.log_to_console:
            print("🚀 LLM 호출 시작...")
    
    def on_llm_end(self, response, **kwargs):
        """LLM 호출 완료 시 실행"""
        # 1. DB에 완료 기록
        self.trace_db.end_run(...)
        
        # 2. 토큰 사용량 기록
        self.trace_db.record_token_usage(...)
        
        # 3. 메트릭 기록
        self.trace_db.record_metric(...)
```

### 4.3 콜백 핸들러 적용

```python
# 콜백 핸들러 인스턴스 생성
monitoring_handler = LocalMonitoringHandler(trace_db)

# LLM에 콜백 적용
llm = ChatOpenAI(
    model="gpt-4o-mini",
    callbacks=[monitoring_handler]  # ← 여기에 추가
)

# 이제 모든 LLM 호출이 자동으로 추적됨
response = llm.invoke("안녕하세요")  # → 자동으로 DB에 기록
```

---

## 5. 메트릭 수집 및 분석

### 5.1 수집되는 메트릭

| 메트릭 | 설명 | 단위 |
|--------|------|------|
| `latency` | 응답 시간 | 초 |
| `prompt_tokens` | 입력 토큰 수 | 개 |
| `completion_tokens` | 출력 토큰 수 | 개 |
| `total_tokens` | 총 토큰 수 | 개 |
| `estimated_cost` | 추정 비용 | USD |
| `success_rate` | 성공률 | % |

### 5.2 비용 추정 로직

```python
def estimate_cost(prompt_tokens, completion_tokens, model="gpt-4o-mini"):
    """토큰 수로 비용 추정"""
    
    # 모델별 가격 (1K 토큰당)
    pricing = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
    }
    
    price = pricing.get(model, pricing["gpt-4o-mini"])
    
    cost = (prompt_tokens / 1000 * price["input"] + 
            completion_tokens / 1000 * price["output"])
    
    return cost
```

### 5.3 SQL 분석 예제

```sql
-- 일별 사용량 통계
SELECT 
    DATE(start_time) as date,
    COUNT(*) as total_runs,
    AVG(duration_seconds) as avg_latency,
    SUM(total_tokens) as total_tokens
FROM runs r
JOIN token_usage t ON r.id = t.run_id
GROUP BY DATE(start_time)
ORDER BY date DESC;

-- 모델별 비용 분석
SELECT 
    model,
    COUNT(*) as calls,
    SUM(estimated_cost) as total_cost,
    AVG(estimated_cost) as avg_cost
FROM token_usage
GROUP BY model;

-- 시간대별 성능 분석
SELECT 
    strftime('%H', start_time) as hour,
    AVG(duration_seconds) as avg_latency,
    COUNT(*) as call_count
FROM runs
WHERE status = 'success'
GROUP BY hour
ORDER BY hour;

-- 에러율 추이
SELECT 
    DATE(start_time) as date,
    ROUND(100.0 * SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) / COUNT(*), 2) as error_rate
FROM runs
GROUP BY DATE(start_time);
```

---

## 6. LangSmith vs 로컬 모니터링

### 6.1 기능 비교표

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    기능 비교표                                          │
├───────────────────────┬─────────────────┬─────────────────────────────┤
│         기능          │  로컬 모니터링   │        LangSmith           │
├───────────────────────┼─────────────────┼─────────────────────────────┤
│ 비용                  │ 무료            │ 유료 (Free tier 있음)       │
│ API 키 필요           │  불필요        │  필요                     │
│ 오프라인 작동         │  가능          │  불가                     │
│ 데이터 저장 위치      │ 로컬 파일        │ 클라우드                    │
│ 프라이버시            │  완전 보장     │ △ 클라우드 저장             │
│ 실시간 대시보드       │ △ 직접 구현      │  제공                     │
│ 팀 협업               │ △ 제한적        │  용이                     │
│ 고급 분석             │ △ SQL 직접 작성 │  내장 기능                │
│ 프롬프트 버전 관리    │  없음          │  Hub 제공                 │
│ 자동화된 평가         │ △ 직접 구현      │  내장                     │
│ 커스터마이징          │  자유로움      │ △ 제한적                   │
└───────────────────────┴─────────────────┴─────────────────────────────┘
```

### 6.2 선택 가이드

```text
▶ 로컬 모니터링 추천:
  ├── 개인 학습/개발 단계
  ├── 비용 절약이 중요한 경우
  ├── 데이터 프라이버시가 중요한 경우
  ├── 오프라인/폐쇄망 환경
  └── 특수한 커스터마이징 필요 시

▶ LangSmith 추천:
  ├── 팀 협업 프로젝트
  ├── 프로덕션 환경
  ├── 고급 분석 및 평가 필요 시
  ├── A/B 테스트 수행 시
  └── 프롬프트 버전 관리 필요 시
```

---

## 7. 적용 가이드

### 7.1 대시보드 구축 (Streamlit 예제)

```python
# dashboard.py
import streamlit as st
import sqlite3
import pandas as pd

st.title(" LLM 모니터링 대시보드")

# DB 연결
conn = sqlite3.connect("local_traces.db")

# 요약 통계
st.header(" 요약")
col1, col2, col3, col4 = st.columns(4)

df_summary = pd.read_sql("""
    SELECT 
        COUNT(*) as total_runs,
        AVG(duration_seconds) as avg_latency,
        SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
    FROM runs
""", conn)

col1.metric("총 실행", df_summary['total_runs'][0])
col2.metric("평균 응답시간", f"{df_summary['avg_latency'][0]:.2f}s")
col3.metric("성공률", f"{df_summary['success_rate'][0]:.1f}%")

# 시간별 추이 차트
st.header(" 응답시간 추이")
df_hourly = pd.read_sql("""
    SELECT datetime(start_time) as time, duration_seconds
    FROM runs WHERE status='success'
    ORDER BY start_time DESC LIMIT 100
""", conn)
st.line_chart(df_hourly.set_index('time'))

# 최근 실행 목록
st.header(" 최근 실행")
df_recent = pd.read_sql("""
    SELECT id, name, duration_seconds, status, start_time
    FROM runs ORDER BY start_time DESC LIMIT 20
""", conn)
st.dataframe(df_recent)
```

### 7.2 알림 시스템 구현

```python
def check_alerts(trace_db):
    """모니터링 알림 체크"""
    summary = trace_db.get_summary()
    
    alerts = []
    
    # 성공률 저하 알림
    if summary['success_rate'] < 95:
        alerts.append(f"⚠️ 성공률 저하: {summary['success_rate']}%")
    
    # 평균 응답시간 초과 알림
    if summary['avg_duration_seconds'] > 5:
        alerts.append(f"⚠️ 응답시간 증가: {summary['avg_duration_seconds']:.2f}s")
    
    # 비용 초과 알림
    if summary['total_estimated_cost'] > 10:
        alerts.append(f"⚠️ 비용 초과: ${summary['total_estimated_cost']:.2f}")
    
    return alerts
```

### 7.3 데이터 내보내기

```python
import pandas as pd
import sqlite3

def export_to_csv(db_path="local_traces.db", output_dir="exports"):
    """모니터링 데이터를 CSV로 내보내기"""
    conn = sqlite3.connect(db_path)
    
    # 실행 기록
    df_runs = pd.read_sql("SELECT * FROM runs", conn)
    df_runs.to_csv(f"{output_dir}/runs.csv", index=False)
    
    # 토큰 사용량
    df_tokens = pd.read_sql("SELECT * FROM token_usage", conn)
    df_tokens.to_csv(f"{output_dir}/token_usage.csv", index=False)
    
    # 메트릭
    df_metrics = pd.read_sql("SELECT * FROM metrics", conn)
    df_metrics.to_csv(f"{output_dir}/metrics.csv", index=False)
    
    conn.close()
```

---

##  요약

###  체크리스트

- [ ] SQLite 추적 시스템 이해
- [ ] 커스텀 콜백 핸들러 구현
- [ ] 메트릭 수집 및 분석
- [ ] SQL 쿼리로 데이터 분석
- [ ] 대시보드 구축 (선택)

### 포인트

1. **API 키 없이도 모니터링 가능** - SQLite + 콜백 핸들러
2. **프라이버시 보장** - 모든 데이터가 로컬에 저장
3. **완전한 커스터마이징** - 원하는 메트릭 추가 가능
4. **SQL로 자유로운 분석** - pandas 연동으로 시각화 가능

###  생성되는 파일

```text
프로젝트 폴더/
├── 06_local_monitoring.py    # 실행 코드
├── local_traces.db           # SQLite 데이터베이스 (자동 생성)
└── exports/                  # CSV 내보내기 (선택)
    ├── runs.csv
    ├── token_usage.csv
    └── metrics.csv
```

---


응용분야

1. **웹 대시보드** - Streamlit/Gradio로 시각화
2. **알림 시스템** - Slack/Email 연동
3. **분산 추적** - 여러 서비스 간 추적
4. **프로덕션 전환** - 필요 시 LangSmith로 마이그레이션
