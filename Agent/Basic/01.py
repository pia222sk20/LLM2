from enum import Enum
#사용 예
# class Status(Enum):
#     PENDING=1
#     RUNNING=2
#     COMPLETED=3
#     FAILED=4

# task_status = Status.RUNNING
# print(task_status)
# print(task_status.name)
# print(task_status.value)

# 상태정의
class AgentState(Enum):
    '''에이전트 상태'''
    IDLE = 'idle'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    ERROR = 'error'

# 데이터 클래스
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

@dataclass
class ExcutionRecord:
    '''실행기록'''
    timestamp:str
    action:str
    status:str
    duration:float
    error:Optional[str] = None

@dataclass
class AgentStats:
    '''에이전트 통계'''
    total_excuted:int = 0
    success_count: int = 0
    error_count:int = 0
    total_time:float = 0.0
    avg_time: float = 0.0

# 에이전트
