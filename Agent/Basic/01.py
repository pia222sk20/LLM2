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
import uuid
from datetime import datetime
class SimpleAgent:
    '''기본에이전트 구현
    - 상태 관리
    - 작업 실행
    - 이력 기록
    - 통계 추적
    '''
    def __init__(self,name:str='Agent', agent_type:str = 'simple'):
        self.angent_id = str(uuid.uuid4())[:8]
        self.name = name
        self.agent_type = agent_type

        self._state = AgentState.IDLE
        self._history : List[ExcutionRecord] = []
        self._stats = AgentStats()
        
        print(f'[SUCCESS] {self.name} 에이전트 생성 (ID : {self.angent_id})')
    # 상태 관리
    def get_state(self) ->str:
        '''현재 상태 반환'''        
        return self._state.name
    def set_state(self, state:AgentState):
        '''현재 상태 변경'''
        self._state = state
        print(f'[STATE] [{self._state.name}] 상태 변경 : {self._state.value}')
    # 작업실행
    def excute(self, input_data:Dict[str, Any]) -> Dict[str,Any]:
        '''작업 실행(메인메소드)'''
        start_time = datetime.now()
        self.set_state(AgentState.PROCESSING)
        try:
            # 입력 검증
            if not self._validate_input(input_data):
                raise ValueError('입력 데이터가 유효하지 않습니다.')
            # 실제 작업 수행
            action = input_data.get('action','unknown')
            result = self._process(action, input_data)
            # 성공 처리
            self.set_state(AgentState.COMPLETED)
            duration = (datetime.now() - start_time).total_seconds()
            
            self._add_to_history(action,'success', duration)
            self._update_stats(success=True,duration=duration)

            return {
                'success' : True,
                'agent_id' : self.angent_id,
                'action' : action,
                'output' : result,
                'duration' : duration
            }

        except Exception as e:
            # 오류처리
            self.set_state(AgentState.ERROR)
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            self._add_to_history(action,'error', duration,error_msg)
            self._update_stats(success=False,duration=duration)
            return {
                'success' : False,
                'agent_id' : self.angent_id,
                'error' : error_msg,                
                'duration' : duration
            }


