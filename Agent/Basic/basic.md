# 에이전트 시스템

## 목차
1. [핵심 개념](#핵심-개념)
2. [구조 설계](#구조-설계)
3. [구현 요소](#구현-요소)
4. [동작 원리](#동작-원리)

---

## 핵심 개념

### 에이전트란?
**에이전트(Agent)** 는 주어진 작업을 자동으로 수행할 수 있는 독립적인 소프트웨어 모듈입니다.

### Basic 단계의 목표
-  단일 에이전트 작동 원리 이해
-  메시지 기반 통신 구현
-  상태 관리 시스템 구축
-  기본 작업 실행 및 결과 반환

---

## 구조 설계

### 아키텍처 다이어그램

![alt text](image.png)

### 주요 컴포넌트

| 컴포넌트 | 역할 | 설명 |
|---------|------|------|
| **Agent** | 중앙 실행 엔진 | 모든 작업 조율 및 실행 |
| **State** | 상태 관리 | IDLE, PROCESSING, COMPLETED 등 |
| **Message** | 통신 | 입출력 데이터 포맷 |
| **History** | 기록 | 실행 이력 저장 |
| **Stats** | 통계 | 성능 지표 추적 |

---

## 구현 요소

### 1. 에이전트 상태 (AgentState)

```
IDLE (대기)
  ↓
PROCESSING (처리)
  ↓
COMPLETED / ERROR (완료/에러)
```

**상태 설명:**
- `IDLE`: 초기 상태, 작업 대기 중
- `PROCESSING`: 작업 진행 중
- `COMPLETED`: 작업 완료
- `ERROR`: 오류 발생

### 2. 메시지 구조

```python
{
    "id": "msg_123",           # 메시지 ID
    "type": "REQUEST",         # 메시지 타입
    "payload": {...},          # 실제 데이터
    "timestamp": "2025-12-15T22:00:00",  # 생성 시간
    "status": "CREATED"        # 메시지 상태
}
```

### 3. 에이전트 실행 흐름

```
입력 받음
   ↓
입력 검증
   ↓
상태 변경 (IDLE → PROCESSING)
   ↓
작업 실행
   ↓
결과 저장
   ↓
상태 변경 (PROCESSING → COMPLETED/ERROR)
   ↓
결과 반환
```

### 4. 실행 기록 (ExecutionRecord)

```python
{
    "timestamp": "2025-12-15T22:00:00",
    "action": "process_image",
    "status": "success",
    "duration": 2.34,          # 실행 시간 (초)
    "input_size": 1024,        # 입력 크기
    "output_size": 2048,       # 출력 크기
    "error": None
}
```

---

## 동작 원리

### 단순 에이전트의 생명주기

```
1. 초기화 (Initialization)
   └─ 에이전트 생성
   └─ 설정 로드
   └─ 상태: IDLE

2. 입력 수신 (Input)
   └─ 사용자 요청 수신
   └─ 입력 검증
   └─ 메시지 생성

3. 처리 (Processing)
   └─ 상태: PROCESSING
   └─ 실제 작업 수행
   └─ 결과 계산

4. 완료 (Completion)
   └─ 상태: COMPLETED
   └─ 결과 저장
   └─ 이력 기록

5. 반환 (Output)
   └─ 결과 반환
   └─ 메타데이터 포함
```

### 코드 예시

```python
# 1. 에이전트 생성
agent = SimpleAgent()

# 2. 작업 실행
result = agent.execute({
    "task": "greet",
    "name": "Alice"
})

# 3. 결과 확인
if result["success"]:
    print(result["output"])
else:
    print(result["error"])
```

---

## Basic 단계 학습 목표

### ✅ 완료해야 할 항목

1. **단일 에이전트 구현**
   - 기본 에이전트 클래스
   - 상태 관리
   - 메시지 처리

2. **작업 실행**
   - 간단한 텍스트 작업
   - 숫자 계산
   - 데이터 변환

3. **결과 관리**
   - 결과 저장
   - 오류 처리
   - 로깅

4. **테스트**
   - 단위 테스트
   - 통합 테스트
   - 성능 테스트

---

## 실행 예시

### 예제 1: 인사 에이전트

```python
agent = GreetingAgent()
result = agent.execute({"name": "Alice"})
# 출력: "Hello, Alice!"
```

### 예제 2: 계산 에이전트

```python
agent = CalculatorAgent()
result = agent.execute({"operation": "add", "a": 5, "b": 3})
# 출력: 8
```

### 예제 3: 텍스트 처리 에이전트

```python
agent = TextProcessorAgent()
result = agent.execute({"text": "HELLO", "operation": "lowercase"})
# 출력: "hello"
```

---

## 정리

| 항목 | 설명 |
|------|------|
| **상태 관리** | 에이전트는 항상 명확한 상태를 가짐 |
| **메시지 기반** | 모든 통신은 메시지 객체를 통해 수행 |
| **기록 유지** | 모든 실행은 기록됨 |
| **오류 처리** | 예외 상황에 대한 명확한 처리 |
| **통계 추적** | 성능 지표를 자동으로 수집 |