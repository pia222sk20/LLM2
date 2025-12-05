# Tavily대신 duckduckgo  웹검색을 RAG에 통합
# 내부검색 + 외부검색 하이브리드
from ddgs import DDGS
import os
import warnings
from typing import List,Literal,TypedDict,Optional
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()


# 필수 라이브러리 로드
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# duckduckgo 검색 클래스 정의
class DuckDuckGoWebSearch():
    def __init__(self,max_results:int = 5, region:str = 'ko-kr'):
        self.max_reesults = max_results
        self.region = region
        self.ddgs = DDGS()
    def search(self, query : str) -> List[Document]:
        # 텍스트 검색
        results = list(self.ddgs.text(
            query,
            region = self.region,
            max_results = self.max_reesults
        ))
        # 결과를 Document형식으로 반환
        documents = []
        for result in results:
            doc = Document(
                page_content=f"{result.get('title','N/A')}\n{result.get('body','N/A')}",
                metadata = {
                    'source' : 'duckduckgo_web_search',
                    'url' : result.get('href','NA'),
                    'title':result.get('title', 'N/A')
                }
            )
            documents.append(doc)
        return documents
    
    def search_news(self, query : str) -> List[Document]:
        # 텍스트 검색
        results = list(self.ddgs.text(
            query,
            region = self.region,
            max_results = self.max_reesults
        ))
        # 결과를 Document형식으로 반환
        documents = []
        for result in results:
            doc = Document(
                page_content=f"{result.get('title','N/A')}\n{result.get('body','N/A')}",
                metadata = {
                    'source' : 'duckduckgo_web_search',
                    'url' : result.get('href','NA'),
                    'title':result.get('title', 'N/A'),
                    'date' :result.get('date','N/A')
                }
            )
            documents.append(doc)
        return documents

# 웹 검색 인스턴스 생성  - 외부문서
web_search = DuckDuckGoWebSearch()

# 내부문서 VectorDB구축
# 내부문서 셈플(회사의 데이터베이스 또는 각종 문서)
internal_documents = [
    Document(
        page_content="""
        우리 회사의 AI 전략은 다음과 같습니다:
        1. 고객 서비스 자동화를 위한 챗봇 도입
        2. 문서 분석을 위한 RAG 시스템 구축
        3. 업무 효율화를 위한 AI 에이전트 개발
        
        현재 LangChain과 OpenAI API를 기반으로 시스템을 구축 중입니다.
        """,
        metadata={"source": "internal", "type": "ai_strategy", "dept": "기술팀"}
    ),
    Document(
        page_content="""
        사내 LLM 사용 가이드라인:
        1. 고객 개인정보는 LLM에 입력하지 않습니다.
        2. 기밀 문서는 승인 후에만 AI 시스템에 연동합니다.
        3. AI 생성 콘텐츠는 반드시 검토 후 사용합니다.
        4. 비용 최적화를 위해 gpt-4o-mini를 우선 사용합니다.
        """,
        metadata={"source": "internal", "type": "guideline", "dept": "보안팀"}
    ),
    Document(
        page_content="""
        RAG 시스템 운영 현황:
        - 구축 일자: 2024년 6월
        - 사용 모델: GPT-4o-mini (답변), text-embedding-3-small (임베딩)
        - VectorDB: ChromaDB (개발), Pinecone (프로덕션)
        - 일일 쿼리 수: 평균 500건
        - 평균 응답 시간: 2.3초
        """,
        metadata={"source": "internal", "type": "operation", "dept": "기술팀"}
    ),
]

# 텍스트 분할(청킹)
text_splitter = RecursiveCharacterTextSplitter(chunk_size= 300, chunk_overlap = 50)
doc_chunks = text_splitter.split_documents(internal_documents)

# VectorDB 구축
vectorStore = Chroma.from_documents(
    documents=doc_chunks,
    embedding=OpenAIEmbeddings(model='text-embedding-3-small'),
    collection_name= 'duckduckgo_rag' 
)

# 내부 문서 리트리버
internal_retriever = vectorStore.as_retriever(search_kwargs={'k':2})

# 하이브리드 RAG 상태 및 노드 정의
class HybridRAGState(TypedDict):
    '''하이브리드 RAG 에이전트 상태'''
    question : str
    internal_docs : List[Document]
    web_docs : List[Document]
    need_web_search : str
    answer : str

class RelevanceGrade(BaseModel):
    '''문서 관련성 평가 결과'''
    binary_score:str = Field(default='no', description='yes or no')

