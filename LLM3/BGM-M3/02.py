# 이론의 코드를 실행 가능한 상태로 구현
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

def check_evnironment():
    '''환경변수 확인'''
    if not os.getenv.get('OPENAI_API_KEY'):
        raise ValueError('check openai key....')
    print('키 확인 완료')

# 임베딩 기본 개념
def embedding_basic():
    '''텍스트를 수치 벡터로 변환하는 임베딩'''
    # openai 임베딩 모델
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

    # 단일 텍스트 임베딩
    text = '한국어 임베딩 테스트입니다.'
    vector = embeddings.embed_query(text)
    print(f'입력 텍스트 : {text}')
    print(f'벡터차원 : {len(vector)}')

    # 여러 텍스트 배치 임베딩
    texts = [
        'LangGraph는 에이전트 프레임워크입니다.',
        'RAG는 검색 증강 생성입니다.',
        ',Python은 프로그래밍 언어입니다.'
    ]
    vectors = embeddings.embed_documents(texts)
    print(f'입력 텍스트 수 : {len(texts)}')
    print(f'벡터차원 : {len(vectors)}')
    print(f'첫번째 벡터차원 : {len(vectors[0])}')
    return embeddings