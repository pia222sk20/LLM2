import os
import warnings
import pickle
from dotenv import load_dotenv

# 경고 메세지 삭제
warnings.filterwarnings('ignore')
load_dotenv()

# openapi key 확인
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('.env 확인.. 키 없음')

# 필수 라이브러리 로드
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

class SimpleRAGSystem:
    '''간단한 RAG 시스탬 래퍼 클래스'''
    def __init__(self,vectorstore, llm, retriver_k=3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_kwargs = {'k' : retriver_k})
        self.chain = self._build_chain()
    def _build_chain(self):
        '''RAG 체인 구성'''
        prompt = ChatPromptTemplate.from_messages([
            ('system','''당신은 제공된 문맥을 바탕으로 질문에 답변하는 AI입니다.
             문맥에 없는 정보는 답변하지 마세요'''),
            ('human','문맥:\n{context}\n\n질문:{question}\n\n답변:')

        ])
        return( 
            {'context':self.retriever|self._fotmat_docs,'question':RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    @staticmethod
    def _fotmat_docs(docs):
        return '\n\n'.join([doc.page_content  for doc in docs ])
    def ask(self,question:str) ->str:
        '''질문에 답변'''
        return self.chain.invoke(question)
    def ask_with_sources(self, question:str)->dict:
        '''질문에 답변 + 출처 반환'''
        answer = self.chain.invoke(question)
        sources = self.retriever.invoke(question)
        return {
            'answer':answer,
            'sources':[ doc.metadata.get('source','unknown') for doc in sources]
        }

if __name__ == '__main__':
    persist_dir = './chroma_db_reg2'
    vectorstore = Chroma(
        persist_directory = persist_dir,
        collection_name = 'persistent_rag',
        embedding_function = OpenAIEmbeddings(model = 'text-embedding-3-small')
    )
    llm = ChatOpenAI( model = 'gpt-4o-mini', temperature=0 )
    rag_system = SimpleRAGSystem(vectorstore, llm)

    print("래퍼 클래스 테스트:")
    result = rag_system.ask_with_sources("VectorDB의 종류를 알려주세요")
    print(f"   질문: VectorDB의 종류를 알려주세요")
    print(f"   답변: {result['answer']}...")
    print(f"   출처: {result['sources']}")