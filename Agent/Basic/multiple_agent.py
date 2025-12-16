# 여러에이전트 협력
# 메세지 기반 통신
# corrdinator를 통한 라우팅
# 3가지 특화 에이전트(Text, Math, Date)
# 비동기 메세지 통신

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import json



#에이전트 상태
class AgentState(Enum):
    IDLE = 'idle'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    ERROR = 'error'

# 데이터 클래스
@dataclass
class Message:
    '''에이전트 간 메세지'''
    message_id:str
    sender_id:str
    receiver_id : Optional[str]
    content : Dict[str,Any]
    timestamp:str
    def to_dict(self) ->Dict[str,Any]:
        return {
            'id':self.message_id,
            'sender' : self.sender_id,
            'receiver' : self.receiver_id,
            'content':self.content,
            'timestamp':self.timestamp
        }
class SpecializedAgent:
    '''특화된 에이전트'''    
    def __init__(self,name:str, speciality:str):
        '''
        Args:
            name : 에이전트이름
            speciality : 전문 분야
        '''
        self.agent_id = str(uuid.uuid4())[:8]
        self.name = name
        self.speciality = speciality
        self._state = AgentState.IDLE
        self._inbox : List[Message] = []
        self._outbox : List[Message] = []
    def receive_message(self, message: Message):
        '''메세지 수신'''
        self._inbox.append(message)
    def send_message(self, receiver_id:str, content:Dict[str, Any]):
        '''메세지 전송'''
        message = Message(
            message_id=str(uuid.uuid4())[:8],
            sender_id= self.agent_id,
            receiver_id=receiver_id,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        self._outbox.append(message)
        return message
    def process_inbox(self) ->list[Dict[str,Any]]:
        '''받은 메세지 처리'''
        self.set_state(AgentState.PROCESSING)
        results = []
        for message in self._inbox:
            result = self._handle_message(message)
            results.append(result)
        
        self._inbox = []  # 처리된 메세지 제거
        self.set_state(AgentState.COMPLETED)
        return results
    def _handle_message(self, message:Message) -> Dict[str,Any]:
        '''메세지 처리(오버라이드 가능)'''
        return {
            'status' : 'handled',
            'message_id' : message.message_id,
            'content' : message.content
        }
    def get_state(self) -> str:
        return self._state.value
    def set_state(self,state:AgentState):
        self._state = state
    def get_info(self)->Dict[str,Any]:
        '''에이전트의 상태를 반환'''
        return {
            'id' : self.agent_id,
            'name' : self.name,
            'specialty' : self.speciality,
            'state' : self.get_state(),
            'inbox_size' : len(self._inbox),
            'outbox_size' : len(self._outbox) 
        }

# 라우터 클래스
class Corrdinator:
    '''에이전트 조정자(메세지 라우터)'''    
    def __init__(self):
        self.agents : Dict[str, SpecializedAgent] = {}   # 에이전트아이디 : 특화된 에이전트
    def register_agent(self, agent:SpecializedAgent):
        '''에이전트 등록'''
        self.agents[agent.agent_id] = agent
    def route_message(self):
        '''모든 에이전트의 메세지를 라우팅'''
        for agent in self.agents.values():  # SpecializedAgent 들...
            for message in agent._outbox:  # 수신자 에이전트의 정보 및 ..
                if message.receiver_id in self.agents:  # validation 체크 모든 에이전트의 아이디중에서 수신자 에이전트가 있으면
                    receiver = self.agents[message.receiver_id]
                    receiver.receive_message(message)
                    print(f'  [OK] {message.message_id} :{agent.name} -> {receiver.name}')
    
    def process_all_agents(self):
        '''모든 에이전트의 메시지를 처리'''
        for agent in self.agents.values():
            agent.process_inbox()   
    def system_status(self):
        '''시스템 상태 출력'''
        for agent in self.agents.values():
            print(agent.get_info() )

# 특화된 에이전트 구현
class TextProcessorAgent(SpecializedAgent):
    '''텍스트 처리 에이전트'''
    def _handle_message(self, message:Message)->Dict[str, Any]:
        content = message.content
        operation = content.get('operation','')
        text = content.get('text','')
        if operation == 'uppercase':
            result = text.upper()
        elif operation == 'lowercase':            
            result = text.lower()
        elif operation == 'reverse':            
            result = text[::-1]
        else:
            result = text

        return {
            'status' : 'processed',
            'operation' : operation,
            'input' : text,
            'output' : result
        }
class MathAgent(SpecializedAgent):
    def _handle_message(self, message:Message)->Dict[str,Any]:
        content = message.content
        operation = content.get('operation','')
        a = content.get('a',0)
        b = content.get('b',0)
        if operation == 'add':
            result = a+b
        elif operation == 'subtract':
            result = a-b
        elif operation == 'multiply':
            result = a*b
        elif operation == 'devide':
            result = a / b if b !=0 else 0
        else:
            result = 0
        return {
            'status' : 'calculated',
            'a' : a,
            'b' : b,
            'result':result
        }
class DataAnalyzerAgent(SpecializedAgent):
    """데이터 분석 에이전트"""
    
    def _handle_message(self, message: Message) -> Dict[str, Any]:
        content = message.content
        data = content.get("data", [])
        
        print(f"   {self.name}: {len(data)}개 항목 분석")
        
        if isinstance(data, list) and len(data) > 0:
            if all(isinstance(x, (int, float)) for x in data):
                avg = sum(data) / len(data)
                max_val = max(data)
                min_val = min(data)
                
                return {
                    "status": "analyzed",
                    "count": len(data),
                    "average": avg,
                    "max": max_val,
                    "min": min_val
                }
        
        return {"status": "invalid_data"}    

if __name__ =='__main__':
    text_agent = TextProcessorAgent('textbot', 'text processing')
    math_agent = MathAgent('mathbot','calcualte')
    analyzer_agent = DataAnalyzerAgent('analyzerbot', 'data analysis')

    # 코디네이터(조정자)에등록
    corrdiantor = Corrdinator()
    corrdiantor.register_agent(text_agent)
    corrdiantor.register_agent(math_agent)
    corrdiantor.register_agent(analyzer_agent)

    # 메세지 생성 및 전송
    text_agent.send_message(
        text_agent.agent_id,
        {'operation':'uppercase','text':'hong-gil-dong'}
    )
    math_agent.send_message(
        math_agent.agent_id,
        {'operation':'multiply','a':10,'b':20}        
    )
    analyzer_agent.send_message(
        analyzer_agent.agent_id,
        {'data':[1,2,3,4,5,6,7,8,9,10]}
    )

    # 메세지 라우팅  에이전트아이디에 해당하는 메세지를 해당 에이전트의 inbox에 저장
    corrdiantor.route_message()  

    # 메세지 처리
    corrdiantor.process_all_agents()

    # 시스템 상태 출력
    corrdiantor.system_status()