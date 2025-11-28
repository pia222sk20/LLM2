# 1. Query Transformation  (질문 변화) - 검색 최적화
# 2. Multi-Query            (다중 질의) - 검색 범위 확대
# 3. Self-RAG               (자기 보정) - 문서 관련성 평가
# 4. Contextual Compressioin (문맥 압축) - 관련 부분만 추출
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
script_dir = os.path.dirname(os.path.abspath(__file__) )
docs_path = os.path.join(script_dir,'advenced','sample_docs', 'langgraph_rag')
print(f'docs paths : {docs_path}')

loader = DirectoryLoader(
    docs_path,
    glob = '**/*.txt',
    loader_cls=TextLoader,
    loader_kwargs={'encoding':'utf-8'}
)
document = loader.load()
print(f'읽은 문서의수 : {len(document)}')
# 텍스트 분할 - 청킹
text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    separators= ['\n\n','\n','.',' ','']
)
# 스플릿 = 청킹
doc_splits =  text_splitter.split_documents(document)
print(f'청킹개수 : {len(doc_splits)}')
# 임베딩 및 VectorDB
embedding_model =  OpenAIEmbeddings(model = 'text-embedding-3-small')

vectorstore =  Chroma.from_documents(
    documents=doc_splits,
    collection_name='basic_rag_collection',
    embedding=embedding_model
)
# 리트리버
base_retriever =  vectorstore.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k' : 3}
)
# LLM 설정
llm = ChatOpenAI(model = 'gpt-4o-mini',temperature=0)
print(f'setup complete!!!!')

# 유틸리티 함수
def format_docs(docs):
    '''문서를 문자열로 포멧팅'''
    return '\n\n---\n\n'.join([ doc.page_content for doc in docs ])

# 질문 재작성 프롬프트
rewrite_prompt=  ChatPromptTemplate.from_template('''
다음 질문을 검색에 더 적합한 형태로 변환해 주세요.
키워드 중심으로 명화기하게 바꿔주세요
변환된 검색어만 출력하세요

원본 질문: {qeustion}
변환된 검색어:
''')

rewrite_chain =  rewrite_prompt | llm | StrOutputParser()

# RAG프롬프트
rag_prompt = ChatPromptTemplate.from_messages([
    ('system','제공된 문맥을 바탕으로 한국어로 답변하세요'),
    ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
])

def query_transformation(question):
    '''Query Transformation  (질문 변화) - 검색 최적화'''
    print(' 1. Query Transformation  (질문 변화) - 검색 최적화')
    print('사용자 질문을 검색에 최적화된 형태로 변환합니다.\n')

    # 1. 질문 변환
    transformed = rewrite_chain.invoke({'qeustion' : question})
    print(f'원본 질문 : {question}')
    print(f'transformed 질문 : {transformed}')
    
    # 2. 변환된 질문으로 검색
    docs = base_retriever.invoke(transformed)
    context = format_docs(docs)
    answer_chain = rag_prompt | llm | StrOutputParser()
    
    answer = answer_chain.invoke({'context':context, 'question':question})
    return answer, [ os.path.basename(d.metadata.get('source','unknown')) for d in docs ]

test_question = [
    'RAG 어떻게 쓰나요?',
    'LangGraph 뭐하는 거야?',
]

for q in test_question:
    print(f'Question : {q}')
    answer, sources = query_transformation(q)
    print(f'answer : {answer}  sources : {sources}')