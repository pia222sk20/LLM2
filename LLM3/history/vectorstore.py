from langchain_redis import RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from config import REDIS_URL, INDEX_NAME
import os

def load_document_and_create_vectorstroe():
    # 문서로드