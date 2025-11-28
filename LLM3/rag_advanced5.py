# 5. Fusion Retrieval       (융합 검색) - 키워 + 벡터 검색 결합
import os
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError('OPENAI_API_KEY not set')

# 필수 라이브러리 로드
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 문서로드
path = 'C:/LLM/LLM3/advenced/sample_docs'
loader = DirectoryLoader(
    path = path,
    glob = '**/*.txt',
    loader_cls = TextLoader,
    loader_kwargs = {'encoding':'utf-8'}
)
docs = loader.load()
# 청크
spliter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    separators= ['\n\n','\n','.',' ','']
)
doc_splits = spliter.split_documents(docs)

embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
# 벡터
vectorstore =  Chroma.from_documents(
    documents=doc_splits,
    collection_name='basic_rag_collection',
    embedding=embedding_model
)
# 리트리버
retriever = vectorstore.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k' : 3}
)
llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)