import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from typing import  List, Tuple
from dotenv import load_dotenv

# langchain
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

#BS25
from rank_bm25 import BM25Okapi
# 환경설정
load_dotenv()

if not os.getenv('OPENAI_API_KEY'):
    raise ValueError('check openai key in .env')

# HuggingFace 임베딩 시도
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except:
    print('pip install langchain-hggingface sentence-transformers')

# 임베딩 모델 정의
class KoreanEmbeddingModels:
    '''한국어 임베딩 모델 팩토리 클래스
    다양한 임베딩 모델을 쉽게 교체할 수 있도록 추상화
    '''
    @staticmethod
    def get_bge_m3(device: str = 'cpu'):
        '''BGE-M3 모델 반환
         - Dense + Sparse 임베딩을 지원
         - 다국어 지원(한국어 우수)
        '''
        return HuggingFaceEmbeddings(
            model_name = 'BAAI/bge-m3',
            model_kwargs = {
                'device':device,
                'trust_remote_code':True
            },
            encode_kwargs = {
                'normalize_embeddings':True,  # 정규화를 해서 코사인 유사도 계산을 용이하게..
                'batch_size':32
            }
        )
    @staticmethod
    def get_multilingual_e5(device:str='cpu'):
        '''Multilingual-E5 모델 반환
        - 경량화
        - 다국어지원
        '''
        return HuggingFaceEmbeddings(
            model_name = 'intfloat/multilingual-e5-large',
            model_kwargs = {'device':device},
            encode_kwargs= {'normalize_embeddings':True}
        )
    @staticmethod
    def get_korean_roberta(device:str = 'cpu'):
        '''BM-K/KoSimCSE-roberta-multitask'''
        return HuggingFaceEmbeddings(
            model_name = 'BM-K/KoSimCSE-roberta-multitask',
            model_kwargs = {'device':device},
            encode_kwargs= {'normalize_embeddings':True}
        )
    @staticmethod
    def get_openai(model:str = 'text-embedding-3-small'):
        '''OpenAI 임베딩 모델'''
        return OpenAIEmbeddings(model=model)

# 데이터 로드
korean_documents = [
    Document(
        page_content="""
        인공지능(AI)은 기계가 인간의 지능을 모방하여 학습하고, 추론하며, 
        문제를 해결할 수 있도록 하는 기술입니다. 최근 대규모 언어 모델(LLM)의 
        발전으로 AI는 자연어 처리, 번역, 요약 등 다양한 분야에서 활용되고 있습니다.
        특히 GPT-4, Claude, Gemini 등의 모델이 주목받고 있습니다.
        """,
        metadata={"source": "ai_intro", "topic": "인공지능"}
    ),
    Document(
        page_content="""
        RAG(Retrieval-Augmented Generation)는 검색 증강 생성 기술로,
        LLM의 한계를 보완합니다. 기업의 내부 문서나 최신 정보를 벡터 
        데이터베이스에 저장하고, 사용자 질문과 관련된 문서를 검색하여
        답변의 정확성을 높입니다. 이를 통해 환각(Hallucination) 현상을 
        줄일 수 있습니다.
        """,
        metadata={"source": "rag_intro", "topic": "RAG"}
    ),
    Document(
        page_content="""
        LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.
        프롬프트 관리, 체인 구성, 메모리 시스템, 에이전트 등
        다양한 기능을 제공합니다. Python과 JavaScript 버전이 있으며,
        OpenAI, Anthropic, Hugging Face 등 다양한 모델과 통합됩니다.
        """,
        metadata={"source": "langchain_intro", "topic": "프레임워크"}
    ),
    Document(
        page_content="""
        벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 검색하는
        데이터베이스입니다. 텍스트, 이미지, 오디오 등을 임베딩 벡터로 
        변환하여 저장하면, 의미적으로 유사한 항목을 빠르게 찾을 수 있습니다.
        ChromaDB, Pinecone, Weaviate, FAISS 등이 대표적입니다.
        """,
        metadata={"source": "vectordb_intro", "topic": "데이터베이스"}
    ),
    Document(
        page_content="""
        한국어 자연어 처리는 영어와 다른 특성을 가집니다. 한국어는 교착어로서
        조사와 어미가 단어에 붙어 문장의 의미를 결정합니다. 따라서 
        형태소 분석, 적절한 토큰화, 다국어 지원 임베딩 모델 사용이 중요합니다.
        KoNLPy, Mecab 등의 한국어 특화 도구를 활용할 수 있습니다.
        """,
        metadata={"source": "korean_nlp", "topic": "한국어"}
    ),
    Document(
        page_content="""
        프롬프트 엔지니어링은 LLM에게 효과적인 지시를 내리는 기술입니다.
        Zero-shot, Few-shot, Chain-of-Thought 등의 기법이 있습니다.
        좋은 프롬프트는 명확하고 구체적이며, 충분한 문맥을 제공해야 합니다.
        시스템 프롬프트를 통해 AI의 역할과 규칙을 정의할 수 있습니다.
        """,
        metadata={"source": "prompt_engineering", "topic": "프롬프트"}
    )
]

test_texts = [
    '한국어 자연어 처리란 무엇인가요',
    'RAG 시스템의 장점을 설명해 주세요',
    '벡터 데이터베이스의 종류'
]
from time import time
# 임베딩 모델 테스트
# openai 임베딩(base)
openai_embeddings = KoreanEmbeddingModels.get_openai()
start_time = time()
openai_vectors = openai_embeddings.embed_documents(test_texts)
elapsed = time() - start_time
print('openai 임베딩')
print(f'벡터차원 : {len(openai_vectors)}')
print(f'처리시간 : {elapsed:.2f}')

# BGE-M3 모델 테스트
hf_embeddings = KoreanEmbeddingModels.get_bge_m3()
start_time = time()
bgem3_vectors = openai_embeddings.embed_documents(test_texts)
elapsed = time() - start_time
print('hf_embeddings 임베딩')
print(f'벡터차원 : {len(bgem3_vectors)}')
print(f'처리시간 : {elapsed:.2f}')

ko_orberta = KoreanEmbeddingModels.get_korean_roberta()
start_time = time()
ko_orberta_vectors = ko_orberta.embed_documents(test_texts)
elapsed = time() - start_time
print('ko_orberta_vectors 임베딩')
print(f'벡터차원 : {len(ko_orberta_vectors)}')
print(f'처리시간 : {elapsed:.2f}')

# VectorDB 구축 , 검색 테스트
# 청킹(텍스트 분할)
text_spliter =  RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
doc_chunks = text_spliter.split_documents(korean_documents)
print(f'문서분할 완료 : {len(doc_chunks)}개 청크')

# VectorDB 구축
vectorStore =  Chroma.from_documents(
    documents=doc_chunks,
    collection_name='korean_docs',
    embedding=hf_embeddings  # BGM-M3 모델
)
print('검색 테스트 결과')
for query in test_texts:
    results = vectorStore.similarity_search(query)
    print(f'\n\n질문 : {query}')
    print(f'검색결과 : {results[0].metadata.get('topic', 'N/A')}')
    print(f'찾은 문장 : {results[0].page_content}')