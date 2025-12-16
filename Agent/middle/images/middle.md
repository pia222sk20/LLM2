# 워크플로우_및_체이닝

단순한 에이전트 실행을 넘어서 **복잡한 워크플로우 구성**, **에이전트 간의 데이터 흐름**, **상태 관리**, 그리고 **시스템 모니터링**을 다룹니다.

### 학습 목표
1. 워크플로우 패턴과 DAG(방향성 비순환 그래프) 이해
2. 에이전트 체인을 통한 순차 처리 구현
3. 실행 상태 저장 및 복원 기능 구현
4. 시스템 모니터링 및 로깅 구현


##  1. 워크플로우 시스템

### 1.1 워크플로우의 개념

워크플로우는 **여러 단계(step)를 순차적으로 실행**하는 시스템입니다. 각 단계는 에이전트에 의해 실행되며, 이전 단계의 출력이 다음 단계의 입력이 됩니다.


![alt text](image.png)


### 1.2 워크플로우의 구성 요소

#### WorkflowStep (단계)
```python
class WorkflowStep:
    - step_id: 단계의 고유 ID
    - agent: 이 단계를 실행할 에이전트
    - input_source: 입력 데이터 소스
    
    execute(input_data) → StepResult
```

#### WorkflowContext (실행 컨텍스트)
```python
class WorkflowContext:
    - workflow_id: 워크플로우 고유 ID
    - current_step: 현재 실행 중인 단계
    - step_results: 각 단계의 실행 결과 저장
    - workflow_state: 워크플로우 상태 (IDLE, RUNNING, COMPLETED, FAILED)
    
    get_step_output(step_id) → Any  # 이전 단계의 결과 조회
```

#### Workflow (워크플로우 관리자)
```python
class Workflow:
    def add_step(step_id, agent) → None
    def execute(initial_input) → Dict
    def print_summary() → None
```

### 1.3 워크플로우 실행 흐름

1. **초기화**: 워크플로우 시작, 초기 데이터 설정
2. **단계 순회**: 등록된 단계를 순차적으로 실행
3. **데이터 변환**: 각 단계에서 입력 → 처리 → 출력
4. **상태 기록**: 각 단계의 결과를 컨텍스트에 저장
5. **에러 처리**: 단계 실패 시 워크플로우 중단
6. **완료**: 모든 단계 완료 후 최종 결과 반환

### 1.4 상태 관리

```python
@dataclass
class StepResult:
    step_id: str
    status: StepState  # PENDING, RUNNING, COMPLETED, FAILED, SKIPPED
    output: Any
    duration: float
    error: Optional[str]
    timestamp: str
```

---

##  2. 에이전트 체인 (Agent Chaining)

### 2.1 체인의 개념

에이전트 체인은 **여러 에이전트를 순차적으로 연결**하여, 이전 에이전트의 출력이 다음 에이전트의 입력이 되도록 하는 패턴입니다.

![alt text](image-1.png)

### 2.2 체인 vs 워크플로우

| 특성 | Chain | Workflow |
|------|-------|----------|
| 용도 | 순차 처리 | 복잡한 프로세스 |
| 데이터 흐름 | 선형 | 복잡한 의존성 |
| 에러 처리 | 간단함 | 정교함 |
| 상태 관리 | 최소 | 포괄적 |
| 복잡도 | 낮음 | 높음 |

### 2.3 AgentChain 구현

```python
class AgentChain:
    def add_agent(agent) → None
    def execute(initial_input) → Dict
    def print_summary() → None

class ChainResult:
    - agent_name: str
    - status: str  # "completed" or "failed"
    - output: Any
    - duration: float
    - error: Optional[str]
```

### 2.4 체인 실행 과정

![alt text](image-2.png)

### 2.5 에러 처리

```python
try:
    output = agent.execute(input_data)
    # 결과 기록
except Exception as e:
    # 에러 기록
    # 체인 중단 또는 대체 경로 실행
```

---

##  3. 상태 지속성 (State Persistence)

### 3.1 상태 저장의 필요성

- **복원성**: 중단된 작업을 재개할 수 있음
- **감시**: 각 단계의 결과를 추적할 수 있음
- **디버깅**: 문제 발생 지점 파악 용이
- **감사**: 실행 이력 기록 및 검증

### 3.2 StateStore 시스템

```python
class StateSnapshot:
    - snapshot_id: str
    - name: str
    - timestamp: str
    - execution_state: Dict[str, Any]
    - metadata: Dict[str, Any]

class StateStore:
    def save_snapshot(name, state) → StateSnapshot
    def load_snapshot(snapshot_id) → StateSnapshot
    def delete_snapshot(snapshot_id) → bool
    def list_snapshots() → List[Dict]
```

### 3.3 스냅샷 저장 전략

#### 1. 단계별 저장
```python
# 각 단계 완료 후 상태 저장
for step in workflow.steps:
    result = step.execute()
    store.save_snapshot(f"step_{step.id}_complete", result)
```

#### 2. 주기적 저장 (Checkpointing)
```python
# 일정 간격으로 전체 상태 저장
if step_count % 5 == 0:
    store.save_snapshot(f"checkpoint_{step_count}", context)
```

#### 3. 선택적 저장
```python
# 중요한 단계에서만 저장
if step.is_critical:
    store.save_snapshot(f"critical_{step.id}", result)
```

### 3.4 ExecutionHistory (실행 이력)

```python
class ExecutionLog:
    - log_id: str
    - workflow_name: str
    - start_time: str
    - end_time: str
    - status: str
    - steps: List[Dict]  # 각 단계의 실행 정보

class ExecutionHistory:
    def create_log(workflow_name) → ExecutionLog
    def add_step_result(log_id, step_name, status, duration)
    def complete_log(log_id, status)
    def get_log(log_id) → ExecutionLog
    def list_logs() → List[Dict]
```

### 3.5 상태 복원 시나리오

```
저장된 상태: {"step": "processing", "data": {...}, "progress": 60}

상태 복원:
1. 이전 상태 로드
2. 마지막 완료된 단계 확인
3. 다음 단계부터 재개
4. 이후 단계 정상 실행
5. 최종 결과 도출
```

---

##  4. 모니터링 및 로깅

### 4.1 로깅 시스템

#### 로그 레벨

```python
class LogLevel(Enum):
    DEBUG = "DEBUG"      # 상세 정보
    INFO = "INFO"        # 일반 정보
    WARNING = "WARNING"  # 경고
    ERROR = "ERROR"      # 오류
    CRITICAL = "CRITICAL" # 심각한 오류
```

#### 로그 항목 구조

```python
@dataclass
class LogEntry:
    timestamp: str       # 로그 생성 시간
    level: str          # 로그 레벨
    component: str      # 로그 출처 컴포넌트
    message: str        # 로그 메시지
    data: Dict[str, Any] # 추가 데이터
```

### 4.2 Logger 구현

```python
class Logger:
    def debug(component, message, data=None) → None
    def info(component, message, data=None) → None
    def warning(component, message, data=None) → None
    def error(component, message, data=None) → None
    def critical(component, message, data=None) → None
    
    def get_logs(level=None, component=None) → List[LogEntry]
    def get_log_summary() → Dict[str, int]
```

### 4.3 메트릭 수집

#### AgentMetrics (에이전트 메트릭)

```python
@dataclass
class AgentMetrics:
    agent_name: str
    executions: int      # 총 실행 수
    successes: int       # 성공 수
    failures: int        # 실패 수
    durations: List[float] # 실행 시간 목록
    
    get_stats() → Dict:
        - success_rate: 성공률 (%)
        - avg_duration: 평균 실행 시간
        - min_duration: 최소 실행 시간
        - max_duration: 최대 실행 시간
        - total_duration: 총 실행 시간
        - stdev_duration: 표준 편차
```

#### WorkflowMetrics (워크플로우 메트릭)

```python
@dataclass
class WorkflowMetrics:
    workflow_name: str
    executions: int       # 총 실행 수
    successes: int        # 성공한 실행
    failures: int         # 실패한 실행
    durations: List[float]
    agent_metrics: Dict[str, AgentMetrics]
    
    get_stats() → Dict:
        - success_rate: 성공률
        - avg_duration: 평균 시간
        - agent_count: 에이전트 수
        - agents: 각 에이전트의 메트릭
```

### 4.4 모니터 (Monitor)

```python
class Monitor:
    def start_workflow(workflow_id, workflow_name) → None
    def end_workflow(workflow_name, duration, success) → None
    def log_agent_execution(workflow_name, agent_name, duration, success) → None
    def add_alert_rule(rule: Callable) → None
    def check_alerts() → None
    def get_metrics(workflow_name) → Dict
    def print_metrics_report() → None
```

### 4.5 경고 규칙 (Alert Rules)

```python
# 실패율이 높을 때
monitor.add_alert_rule(lambda stats: 
    stats['failure_rate'] > 30)

# 실행 시간이 오래 걸릴 때
monitor.add_alert_rule(lambda stats: 
    stats['avg_duration'] > 5.0)

# 특정 에이전트가 자주 실패할 때
monitor.add_alert_rule(lambda stats:
    any(agent['success_rate'] < 50 for agent in stats['agents'].values()))
```

### 4.6 모니터링 리포트 예시

```
 모니터링 메트릭 리포트
=================================================

워크플로우: 데이터_처리
  실행: 10 | 성공: 9 | 실패: 1
  성공률: 90.0%
  평균 시간: 2.5초
  
   에이전트 메트릭:
    - Agent1: 실행 10회, 성공률 100%
    - Agent2: 실행 10회, 성공률 90%
    - Agent3: 실행 10회, 성공률 100%
```

---

##  5. 중급 단계 아키텍처

### 5.1 전체 시스템 구조

![alt text](image-3.png)


### 5.2 데이터 흐름

```
Input Data
  ↓
Workflow/Chain Execute
  ↓
StateStore Save (상태 저장)
  ↓
Logger Record (로그 기록)
  ↓
Monitor Metrics (메트릭 수집)
  ↓
Output Data
```

### 5.3 에러 처리 흐름

```
Exception Occurred
  ↓
Logger.error() (에러 로그)
  ↓
Monitor.log_agent_execution(success=False) (메트릭 업데이트)
  ↓
StateStore Save (에러 상태 저장)
  ↓
Alert Check (경고 확인)
  ↓
Workflow/Chain Fail (프로세스 중단)
  ↓
Return Error Result
```

---

##  6. 성능 최적화

### 6.1 체인 최적화

```python
#  비효율적: 모든 단계를 항상 실행
for agent in agents:
    result = agent.execute(data)

#  효율적: 조건부 실행
for agent in agents:
    if should_execute(agent):
        result = agent.execute(data)
```

### 6.2 상태 저장 최적화

```python
#  비효율적: 매번 저장
for step in steps:
    result = step.execute()
    store.save_snapshot(f"step_{i}", result)  # 매번 저장

#  효율적: 선택적 저장
for step in steps:
    result = step.execute()
    if step.is_critical or i % 5 == 0:
        store.save_snapshot(f"step_{i}", result)
```

### 6.3 로그 관리

```python
#  비효율적: 모든 DEBUG 로그 저장
logger.debug("component", "메시지")

#  효율적: 필요한 정보만 저장
if config.debug_enabled:
    logger.debug("component", "메시지")
```

---

## 7. 디버깅 및 문제 해결

### 7.1 일반적인 문제

#### 문제 1: 데이터 타입 불일치
```python
#  에러: Agent2는 list를 예상하지만 dict를 받음
agent1.output = {"data": [1, 2, 3]}  # dict 반환
agent2.input = agent1.output  # dict 입력

#  해결: 타입 검증
def validate_input(data, expected_type):
    if not isinstance(data, expected_type):
        raise TypeError(f"Expected {expected_type}, got {type(data)}")
```

#### 문제 2: 상태 불일치
```python
#  문제: 저장된 상태와 실제 상태가 다름
saved_state = {"step": 3, "data": "old_data"}
actual_state = {"step": 5, "data": "new_data"}

#  해결: 상태 검증 및 동기화
def sync_states(saved_state, actual_state):
    if saved_state != actual_state:
        log.warning("상태 불일치 감지")
        return actual_state  # 실제 상태 우선
```

#### 문제 3: 에러 전파
```python
#  문제: 에러가 전파되지 않음
try:
    result = agent.execute(data)
except Exception as e:
    pass  # 에러 무시

#  해결: 에러 처리 및 기록
try:
    result = agent.execute(data)
except Exception as e:
    logger.error("Agent", f"실행 실패: {e}")
    raise  # 에러 전파
```

### 7.2 디버깅 팁

1. **상세 로깅 활성화**: DEBUG 레벨에서 로그 수집
2. **상태 덤프**: 각 단계의 상태를 파일에 저장
3. **메트릭 분석**: 성능 병목 지점 파악
4. **재현**: 저장된 상태에서 다시 실행

---

##  8. 실습 예제

###  텍스트 처리 워크플로우

```python
# 워크플로우 생성
workflow = Workflow("텍스트 처리")

# 단계 추가
workflow.add_step("load", DataLoadingAgent())
workflow.add_step("normalize", TextNormalizerAgent())
workflow.add_step("analyze", TextAnalyzerAgent())

# 실행
result = workflow.execute()
workflow.print_summary()
```

### 데이터 처리 체인

```python
# 체인 생성
chain = AgentChain("데이터 처리")

# 에이전트 추가
chain.add_agent(DataInputAgent())
chain.add_agent(DataValidationAgent())
chain.add_agent(DataTransformAgent())
chain.add_agent(DataOutputAgent())

# 실행
result = chain.execute()
chain.print_summary()
```

### 예제 3: 모니터링과 함께 실행

```python
monitor = Monitor()

# 경고 규칙 추가
monitor.add_alert_rule(lambda stats: stats['success_rate'] < 80)

# 워크플로우 실행
monitor.start_workflow("wf001", "데이터 처리")
# ... 워크플로우 실행 ...
monitor.end_workflow("데이터 처리", 2.5, success=True)

# 메트릭 출력
monitor.print_metrics_report()
```

---

##  9. 주요 개념 정리

| 개념 | 설명 | 사용 시기 |
|------|------|----------|
| **Workflow** | 복잡한 다단계 프로세스 | 복잡한 비즈니스 로직 |
| **AgentChain** | 순차적인 에이전트 연결 | 단순 파이프라인 |
| **StateStore** | 상태 저장 및 복원 | 장시간 실행, 재개 기능 필요 |
| **Logger** | 상세 로그 기록 | 디버깅, 감시 |
| **Monitor** | 메트릭 수집 및 분석 | 성능 모니터링 |


