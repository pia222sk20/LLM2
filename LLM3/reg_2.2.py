import os 
from dotenv import load_dotenv
import warnings
import pickle

warnings.filterwarnings('ignore')
load_dotenv()

# 필수 라이브러리 로드
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

# 이전 단계 데이터 로드 또는 재생성
print('청크데이터 로드')
chunks_file = 'chunks_output.pkl'

if os.path.exists(chunks_file):
    with open(chunks_file, 'rb') as f:
        doc_chunks = pickle.load(f)
else:
    #파일이 없으면 새로 생성
    print('reg_1.2.py  실행')

# openai 임베딩 모델 초기화
embedding_model = OpenAIEmbeddings(
    model = 'text-embedding-3-small'
)

# 단일 텍스트 임베딩 테스트
test_text = 'RAG는 검색 증강 생성 기술입니다.'

start_time = time.time()
test_embedding = embedding_model.embed_query(test_text)
elapsed = time.time() - start_time

print(f'입력텍스트 : {test_text}')
print(f'백터차원 : {len(test_embedding)}')
print(f'백터일부 : {test_embedding[0]:.4f} {test_embedding[1]:.4f} ... {test_embedding[-1]:.4f}')
print(f'소요시간 : {elapsed:.3f}')

# 유사도 계산
import numpy as np
def cosine_similarity(vec1,vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1*norm2)

test_sentences = [
    "RAG는 검색 증강 생성 기술입니다.",         # 기준 문장
    "RAG는 문서 검색과 답변 생성을 결합합니다.",  # 유사한 문장
    "벡터 데이터베이스는 임베딩을 저장합니다.",   # 관련 있는 문장
    "오늘 날씨가 매우 좋습니다.",               # 관련 없는 문장
]
# 모든 문장을 임베딩
embeddings = [ embedding_model.embed_query(sent) for sent in test_sentences ]
# 기존 문장과 유사도 비교
base_embedding = embeddings[0]
print(f'기준 문장 : {test_sentences[0]}')
print(f'유사도 비교 결과')
for i, (sent, emb) in enumerate(zip(test_sentences[1:],embeddings[1:]), 1):
    similarity = cosine_similarity(base_embedding,emb)
    print(f' {i} {sent[:30]} --> {similarity:.4f}')

# ChromaDB  백터DB
# 동작방식
# 저장 : 텍스트(청크) -> 임베딩(백터) -> VectorDB(저장)
# 검색 : 질문 -> 임베딩(백터) -> 유사도검색 -> top-k문서 반환

# chromaDB에 청크 저장
start_time = time.time()
# chromaDB 생성(인메모리)
vectorstore = Chroma.from_documents(
    document = doc_chunks,
    collection_name = 'reg_2.2',
    embedding = embedding_model
)
elapsed = time.time() - start_time
print(f'VectorDB 구축 완료')
print(f'저장된 청크수 : {len(doc_chunks)}')
print(f'소요시간 : {elapsed:.2f}')