import os
import warnings
warnings.filterwarnings("ignore")

from typing import List, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END

# 환경설정
load_dotenv()

if not os.environ.get('OPENAI_API_KEY'):
    raise ValueError('key check....')

# state 정의
# TypeDict 상태 스키마 정의
class RAGState(TypedDict):
    '''RAG 에이전트의 상태 스키마'''
    question : str
    documents: List[Document]
    context : str
    answer : str

initial_state : RAGState = {
    'question' : 'RAG란 무엇인가요?',
    'documents' : [],
    'context':'',
    'answer':''
}    
print(f'초기상태 : {initial_state}')

# 상태 업데이트(시뮬레이션)
state = initial_state.copy()

# 1. 검색 노드가 문서를 추가
state['documents'] = [
    Document(page_content='RAG는 검색 증강 생성입니다.', metadata={'source':'wiki'}),
    Document(page_content='RAG는 LLM의 한계를 극복합니다.', metadata={'source':'blog'}),
]

# 2. 생성 노드가 답변을 생성
state['context'] = '\n'.join([ doc.page_content for doc in state['documents']])
state['answer'] = 'RAG는 검색 증강 생성 기술입니다.'

# node 함수 정의
# 노드는 state를 입력받아서 dict를 반환
# 반환된 dict가 state와 병합

class SimpleState(TypedDict):
    '''단순화된 형태'''
    question : str
    document : List[Document]
    answer: str

# 노드 함수 1 : 검색 노드
def retrieve_node(state:SimpleState)->dict:
    '''검색노드 : 질문을 기반으로 관련 문서를 검색'''
    question = state['question']
    print(f'검색노드 실행 : {question}')
    # 시뮬레이션 : 실제는 retriever.invoke(question) 사용
    mock_documents = [
        Document(page_content='', metadata={}),
        Document(page_content='', metadata={})
    ]
    return {'document':mock_documents}