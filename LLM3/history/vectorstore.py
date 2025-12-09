from langchain_redis import RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from config import REDIS_URL, INDEX_NAME
import os

def create_vectorstore(filepath:str = 'documents/sample.txt'):
    # 문서로드
    loader = TextLoader(filepath,encoding='utf-8')
    docs = loader.load()
    # 청크단위로 분리
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap=50
            )
    chuncks = splitter.split_documents(docs)
    # 임베딩 모델
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore =  RedisVectorStore.from_documents(
        documents=chuncks,
        embedding=embeddings,
        redis_url=REDIS_URL,
        index_name=INDEX_NAME
    )
    return vectorstore