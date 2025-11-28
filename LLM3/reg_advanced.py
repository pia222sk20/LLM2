# 1. Query Transformation  (질문 변화) - 검색 최적화
# 2. Multi-Query            (다중 질의) - 검색 범위 확대


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

# for q in test_question:
#     print(f'Question : {q}')
#     answer, sources = query_transformation(q)
#     print(f'answer : {answer}  sources : {sources}')

# 2. Multi-Query            (다중 질의) - 검색 범위 확대
# 다중 쿼리 생성 프롬프트
multi_query_prompt =  ChatPromptTemplate.from_template('''
다음 질문에 대해 3가지 다른 관점의 검색 쿼리를 생성하세요.
각 쿼리는 새 줄로 구분하여 출력하세요
번호나 설명 없이 쿼리만 출력하세요
                                                       
원본 질문 : {question}
다른 관점의 쿼리들:
''')
# lag chain 구성  LCEL
multi_query_chain = multi_query_prompt | llm | StrOutputParser()

def multi_query_rag(question):
    '''다중 쿼리로 검색해서 결과 통일'''
    # 1.다중 쿼리 생성
    queries_text = multi_query_chain.invoke( {'question': question} )
    queries = [ q.strip() for q in  queries_text.strip().split('\n') if q.strip()]
    # 각 쿼리(질문)으로 검색하고 결과를 통합 (중복제거)
    all_docs = []
    seen_contents = set()
    for query in queries:
        docs = base_retriever.invoke(query)
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                all_docs.append(doc)
    print(f'검색된 문서의 개수 : {len(all_docs)}')
    # 리트리버 답변 생성 추출된 문서의 상위 3개만 사용
    context = format_docs(all_docs[:3])
    answer_chain = rag_prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({'context' : context, 'question':question})
    return answer, [ os.path.basename(d.metadata.get('source','unknown')) for d in all_docs ]
     
# 테스트
test = [
    'LangChain 시작하는 방법'
]
for q in test:
    print(f'question : {q}')
    answer, sources = multi_query_rag(q)
    print(f'answer : {answer}')
    print(f'answer : {sources}')

