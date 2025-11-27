# 임베딩 : 텍스트나 이미지를 고정길이의 숫자 벡터로 변환
# 의미적 유사성을 거리로 표현
# 임베딩 차원
    # 낮은 차원 384     빠름, 저장효율성    all-MiniLM-L6
    # 중간 차원 1024    균형잡힌 선택       BGE-M3, E5
    # 높은 차원 1536+   높은 표현력         OpenAI text-embedding-3
# 주요 임베딩 모델
    # text-embedding-3-samll    한국어 가능
    # text-embedding-3-large    한국어 가능
    # BGE-M3                    한국어 우수
    # KoSimCSE                  한국어 전용

# openai , 허깅페이스(오픈소스)
import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
vector = embeddings.embed_query('안녕하세요')
print(len(vector))

print('-'*100)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-m3',
    # model_kwargs = {'device':'cuda'}
)
vector = embeddings.embed_query('안녕하세요')
print(len(vector))

# VectorDB : 
from langchain_chroma import Chrorma
from langchain_openai import OpenAIEmbeddings
# 임베딩 모델 설정
embeddings = OpenAIEmbeddings(model ='text-embdding-3-small')

# 인메모리 VectorDB 생성
# vectorstore = Chroma.from_documents(
#     documents = doc_chunks
#     embedding=embeddings
#     collection_name = "my_collection"
# )
