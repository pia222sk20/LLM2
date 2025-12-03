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
    if not os.getenv('OPENAI_API_KEY'):
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

def cosine_simularity():
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """두 벡터의 코사인 유사도 계산"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)
        
    # 임베딩 모델
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 테스트 문장들
    sentences = [
        "나는 행복합니다.",           # 기준 문장
        "나는 기쁩니다.",             # 유사한 의미
        "오늘 날씨가 좋습니다.",       # 다른 주제
        "I am happy.",              # 영어 번역
    ]

    # 임베딩 생성
    vectors = [embeddings.embed_query(s) for s in sentences]

    print("\n[코사인 유사도 비교]")
    print(f"   기준 문장: '{sentences[0]}'")
    print()

    base_vector = vectors[0]
    for i, (sentence, vector) in enumerate(zip(sentences[1:], vectors[1:]), 1):
        similarity = cosine_similarity(base_vector, vector)
        print(f"   vs '{sentence}': {similarity:.4f}")

    print("   - 유사도 1.0: 완전히 동일")
    print("   - 유사도 0.8+: 매우 유사")
    print("   - 유사도 0.5+: 어느 정도 관련")
    print("   - 유사도 0.3-: 거의 무관")

    print("\n 코사인 유사도 계산 완료!")
    return cosine_similarity

# BM25 Sparse 검색  희소백터   키워드 중심의 검색(일치성)
def bm25_sparse_search():
    '''키워드 기반의 Sparse 검색을 구현'''
    # 문서 데이터
    documents = [
        "LangGraph는 LangChain 위에 구축된 에이전트 프레임워크입니다.",
        "RAG는 Retrieval-Augmented Generation의 약자입니다.",
        "Python은 데이터 과학에서 가장 많이 사용되는 언어입니다.",
        "벡터 데이터베이스는 임베딩을 저장하고 검색합니다.",
        "ChromaDB는 오픈소스 벡터 데이터베이스입니다.",
    ]
    # 간단한 한국어 토큰화(공백 + 조사 분리)
    def simple_korean_tokenize(text:str) ->List[str]:
        '''공백으로 분리 후 각 단어를 2-gram으로 분리'''
        tokens = []
        for word in text.split():
            tokens.append(word)
            # 2글자 이상이면 n-gram도 추가
            if len(word) >=2:
                for i in range(len(word) -1):
                    tokens.append(word[i:i+2])
        return tokens
    # 문서 토큰화
    tokenized_docs = [simple_korean_tokenize(doc) for doc in documents]
    print(f'토큰화....')
    print(f'원본[0] : {documents[0][:30]}...')
    print(f'토큰[0] : {tokenized_docs[0][:10]}...')

    # bm25 인덱스 생성
    bm25 =  BM25Okapi(tokenized_docs)

    # 검색 테스트
    quries = [
        'LangChain 프레임워크',
        '벡터 데이터베이스',
        'Python 프로그래밍'
    ]
    print('\n[BM25 검색 결과]')
    for query in quries:
        tokenized_query = simple_korean_tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        # 상위 2개 결과
        top_indices = np.argsort(scores)[::-1][:2]
        print(f'\n  질문 : {query}')
        for idx in top_indices:
            print(f'    {scores[idx]:.2f}  {documents[idx][:40]}...')
    return bm25,documents,simple_korean_tokenize



# 하이브리드 검색 구현
def hybrid_search():
    '''Dense(의미 기반)와 Sparse(키워드 기반) 검색을 결합'''

    # 문서데이터 (실제는 전용 Loader를 사용(예 TextLoader  PdfLoader 등) 또는 사용자가 직접 수집한 데이터를 Document 객체로 만들어서 리스트형태)
    documents = [
        Document(page_content="LangGraph는 LangChain 위에 구축된 상태 기반 에이전트 프레임워크입니다.", metadata={"id": 1}),
        Document(page_content="RAG(Retrieval-Augmented Generation)는 검색과 생성을 결합한 기술입니다.", metadata={"id": 2}),
        Document(page_content="Python은 데이터 과학과 AI 개발에 널리 사용됩니다.", metadata={"id": 3}),
        Document(page_content="벡터 임베딩은 텍스트를 수치 벡터로 변환합니다.", metadata={"id": 4}),
        Document(page_content="ChromaDB는 로컬에서 사용할 수 있는 벡터 데이터베이스입니다.", metadata={"id": 5}),
    ]

    # Dense 검색 설정 - BGM-M3 오픈소스로도 변경 가능
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name='hybrid_example'
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={'k':3})

    # Sparse 검색 설정(BM25)
    # 간단한 한국어 토큰화(공백 + 조사 분리)
    def simple_korean_tokenize(text:str) ->List[str]:
        '''공백으로 분리 후 각 단어를 2-gram으로 분리'''
        tokens = []
        for word in text.split():
            tokens.append(word)
            # 2글자 이상이면 n-gram도 추가
            if len(word) >=2:
                for i in range(len(word) -1):
                    tokens.append(word[i:i+2])
        return tokens
    doc_texts =  [ doc.page_content for doc in documents]
    tokenized_docs = [  simple_korean_tokenize(text) for text in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)
    
    def sparse_search(query:str, k:int=3)->List[Document]:        
        '''BM25 기반 검색'''        
        tokenized_query = simple_korean_tokenize(query)
        scores = bm25.get_scores(tokenized_query)        
        top_indices = np.argsort(scores)[::-1][:k]
        return [documents[i] for i in top_indices]
    
    # 하이브리드 검색
    def hybrid_search(query:str, dense_weight: float = 0.7, k:int=3) -> List[Document]:
        '''Dense + Sparse 하이브리드 검색
        Args:
            query : 검색
            dense_weight : Dense 검색 가중치
            k : 반환할 문서 수
        '''
        sparse_weight = 1 - dense_weight
        # Dense 검색
        dense_result = dense_retriever.invoke(query)
        # Sparse 검색
        sparse_results = sparse_search(query, k=k)
        # RRF(Reciprocal Rank Fusion) 점수 계산
        doc_scores = {}
        for rank,doc in enumerate(dense_result):
            doc_id = doc.metadata.get('id')
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc':doc, 'score':0}
            doc_scores[doc_id]['score'] += dense_weight + (1/(rank+1))
        for rank,doc in enumerate(sparse_results):
            doc_id = doc.metadata.get('id')
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc':doc, 'score':0}
            doc_scores[doc_id]['score'] += sparse_weight + (1/(rank+1))
        # 점수순 정렬
        sorted_results = sorted(  doc_scores.values(), key = lambda x: x['score'], reverse=True  )
        return [ item["doc"] for item in sorted_results[:k]]
    # --- 테스트 ---
    test_queries = [
        "에이전트 프레임워크",     # Dense에 유리 (의미)
        "ChromaDB",              # Sparse에 유리 (정확한 키워드)
        "RAG 검색 생성"          # 하이브리드에 유리
    ]
    print("\n[검색 방식별 비교]")
    for query in test_queries:
        print(f"\n 질문: '{query}'")
        
        # Dense만
        dense_results = dense_retriever.invoke(query)
        print(f"   [Dense] {dense_results[0].page_content[:40]}...")
        
        # Sparse만
        sparse_results = sparse_search(query, k=1)
        print(f"   [Sparse] {sparse_results[0].page_content[:40]}...")
        
        # 하이브리드
        hybrid_results = hybrid_search(query, dense_weight=0.7, k=1)
        print(f"   [Hybrid] {hybrid_results[0].page_content[:40]}...")
    
    # 정리
    vectorstore.delete_collection()
    
    print("\n 하이브리드 검색 완료!")
    return hybrid_search



if __name__ == '__main__':
    check_evnironment()
    # embedding_basic()
    # cosine_simularity()
    # bm25_sparse_search()
    hybrid_search()