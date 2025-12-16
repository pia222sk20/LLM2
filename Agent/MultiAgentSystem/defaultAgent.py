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
class Coordinator:
    def __init__(self):
        self.agents: Dict[str, SpecializedAgent] = {}
    
    def register_agent(self, agent: SpecializedAgent):
        self.agents[agent.agent_id] = agent
    
    def route_message(self):
        for agent in self.agents.values():
            for message in agent._outbox:
                if message.receiver_id in self.agents:
                    receiver = self.agents[message.receiver_id]
                    receiver.receive_message(message)
                    print(f'  ✓ {message.message_id}: {agent.name} → {receiver.name}')
            agent._outbox = []
    
    def process_all_agents(self):
        for agent in self.agents.values():
            if agent._inbox:
                agent.process_inbox()