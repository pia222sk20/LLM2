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
    total_executed:int = 0
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
        self.agent_id = str(uuid.uuid4())[:8]
        self.name = name
        self.agent_type = agent_type

        self._state = AgentState.IDLE
        self._history : List[ExcutionRecord] = []
        self._stats = AgentStats()
        
        print(f'[SUCCESS] {self.name} 에이전트 생성 (ID : {self.agent_id})')
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
                'agent_id' : self.agent_id,
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
                'agent_id' : self.agent_id,
                'error' : error_msg,                
                'duration' : duration
            }
    # 작업 처리
    def _process(self, action:str, input_data:Dict[str,Any]) -> Any:
        '''실제 작업수행'''
        if action == 'greet':
            name = input_data.get('name', 'Guest')
            return f'Hello, {name}'
        elif action == 'add':
            a = input_data.get('a', 0)
            b = input_data.get('b', 0)
            print(f' 계산 : {a} + {b} = {a+b}')
            return a + b
        elif action == 'multiply':
            a = input_data.get('a', 1)
            b = input_data.get('b', 1)
            print(f' 계산 : {a} x {b} = {a*b}')
            return a * b
        elif action =='uppercase':
            text = input_data.get('text','')
            result = text.upper()
            print(f' 변환 : {text} -> {result}')
            return result
        elif action == 'lowercase':
            text = input_data.get('text','')
            result = text.lower()
            print(f' 변환 : {text} -> {result}')
            return result
        else:
            raise ValueError(f'알수 없는 작업 : {action}')
    # 검증 및 기록
    def _validate_input(self, input_data:Dict[str,Any]) -> bool:
        '''입력 검증'''
        if not isinstance(input_data, dict):
            print('[ERROR] 입력은 딕셔너리여야 합니다.')
            return False
        if 'action' not in input_data:
            print('[ERROR] action 키는 필수입니다..')
            return False
        return True
    def _add_to_history(self, action:str , status:str, duration:float, error:Optional[str] = None):
        '''이력추가'''
        record = ExcutionRecord(
            timestamp=datetime.now().isoformat(),
            action=action,
            status=status,
            duration=duration,
            error=error
        )
        self._history.append(record)
        if len(self._history) > 100:  # 최대 100개만 유지
            self._history.pop(0)
    def _update_stats(self, success:bool, duration:float):
        '''통계업데이트'''
        self._stats.total_executed += 1
        if success:
            self._stats.success_count += 1
        else:
            self._stats.error_count += 1
        self._stats.total_time += duration
        self._stats.avg_time = self._stats.total_time / self._stats.total_executed
    # 정보조회
    def get_info(self) -> Dict[str, Any]:
        """에이전트 정보 반환"""
        return {
            "id": self.agent_id,
            "name": self.name,
            "type": self.agent_type,
            "state": self.get_state(),
            "history_size": len(self._history),
            "stats": asdict(self._stats)
        }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """실행 이력 반환"""
        return [asdict(record) for record in self._history]
    
    def print_info(self):
        """에이전트 정보 출력"""
        info = self.get_info()
        print("\n" + "="*60)
        print(f"[INFO] {self.name} 에이전트 정보")
        print("="*60)
        print(f"ID: {info['id']}")
        print(f"타입: {info['type']}")
        print(f"상태: {info['state']}")
        print(f"이력 개수: {info['history_size']}")
        print(f"\n[STATS] 통계:")
        stats = info['stats']
        print(f"  - 총 실행: {stats['total_executed']}회")
        print(f"  - 성공: {stats['success_count']}회")
        print(f"  - 실패: {stats['error_count']}회")
        print(f"  - 평균 시간: {stats['avg_time']:.3f}초")
        print("="*60 + "\n")

if __name__ == '__main__':
    # 에이전트 생성
    agent = SimpleAgent('Worker','simple')
    test_case = [
        {'action' : 'greet', 'name':'hong-gil-dong'},
        {'action' : 'add', 'a':10,'b':20},
        {'action' : 'uppercase', 'text':'hello-world'},
        {'action' : 'count_words', 'text':'hello my name is hong'},
            ]
    for task in test_case:
        result = agent.excute(task)        
    # 에이전트 정보 출력
    agent.print_info()

    # 이력 출력 - 최근 5개
    history = agent.get_history()
    for his in history[:5]:
        print(his) 


