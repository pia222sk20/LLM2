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

# 최종 답변
rag_prompt = ChatPromptTemplate.from_messages([
    ('system','제공된 문맥을 바탕으로 한국어로 답변하세요'),
    ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
])

# 융합검색
# 1 개별 검색 결과 비교
# 2 융합 결과로 답변 생성

from langchain_community.retrievers import BM25Retriever
# BM25 리트리버         : 키워드기반
# Vector 리트리버       : 의미기반
bm25_retriever = BM25Retriever.from_documents(doc_splits)
bm25_retriever.k = 3

question = 'VectorDB의 종류를 알려주세요'
 # 백터 검색
vector_docs = retriever.invoke(question)
# BM25 검색
bm25_docs = bm25_retriever.invoke(question)
fusion_scores = {}
# 백터 검색 결과 점수
for rank, doc in enumerate(vector_docs):
    doc_key = doc.page_content[:50]
    score = 1 / (60 + rank)
    fusion_scores[doc_key] = fusion_scores.get(doc_key,0) + score
# BM25 검색 결과 점수
for rank, doc in enumerate(bm25_docs):
    doc_key = doc.page_content[:50]
    score = 1 / (60 + rank)
    fusion_scores[doc_key] = fusion_scores.get(doc_key,0) + score
# 점수로 정렬
sorted_docs =  sorted(
    fusion_scores.items(), key=lambda x : x[1], reverse=True
)

print(f'fusion docs 결과 : {sorted_docs}')



