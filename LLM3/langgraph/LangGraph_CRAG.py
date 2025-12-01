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
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END

# 환경설정
load_dotenv()

if not os.environ.get('OPENAI_API_KEY'):
    raise ValueError('key check....')

class CGRAState(TypedDict):
    question : str
    documents : List[Document]
    web_search_needed : str   # 웹검색 여부(yes / no)
    context : str
    answer : str
    grade_results : List[str]   #각 문서의 평가 결과

# 문서
path = 'C:/LLM/LLM3/advenced/sample_docs'
loader = DirectoryLoader(
    path = path,
    glob = '**/*.txt',
    loader_cls = TextLoader,
    loader_kwargs = {'encoding':'utf-8'},        
)
docs = loader.load()

# 텍스트 분할

# 임베딩 및 VectorDB

# 리트리버 설정
