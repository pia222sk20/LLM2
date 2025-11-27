# LangChain 프롬프트 템플릿
# LCEL 사용법
# RAG 체인 구성 및 실행
# 답변 품질 개선 전략

# 파이프라인
# [질문] - > [리트리버] - >[관련문서] ->[프롬프트] -> [LLM] -> [답변]

import os
import warnings
import pickle
from dotenv import load_dotenv

# 경고 메세지 삭제
warnings.filterwarnings('ignore')
load_dotenv()

# openapi key 확인
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('.env 확인.. 키 없음')

# 필수 라이브러리 로드
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

# vectorDB 로드
# 임베딩 모델 초기화
embedding_model = OpenAIEmbeddings(model = 'text-embedding-3-small')
# 이전 단계에서 저장한 vectordb 로드
persist_dir = './chroma_db_reg2'
config_file = 'vectordb_config.pk'
if os.path.exists(persist_dir):
    vectorstore = Chroma(
        persist_directory = persist_dir,
        collection_name = 'persistent_rag',
        embedding_function = embedding_model
    )
else:
    raise ValueError('이전단계 chroma_db_reg2 디렉터리 생성 필요')

# 리트리버 생성
retriever =  vectorstore.as_retriever(search_kwargs={'k':3})

# LLM 모델 생성
llm = ChatOpenAI(
    model = 'gpt-4o-mini',
    temperature=0
)

# 프롬프트 템플릿
# 기본 RAG 프롬프트
basic_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 제공된 문맥(Context)을 바탕으로 질문에 답변하는 AI 어시스턴트입니다.

규칙:
1. 제공된 문맥 내의 정보만을 사용하여 답변하세요.
2. 문맥에 없는 정보는 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요.
3. 답변은 한국어로 명확하고 간결하게 작성하세요.
4. 가능하면 구조화된 형태(목록, 번호 등)로 답변하세요."""),
    ("human", """문맥(Context):
{context}

질문: {question}

답변:""")
])

# 상세 RAG 프롬프트
detailed_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 전문적인 지식 기반 Q&A 시스템입니다.

## 역할
제공된 문맥을 분석하여 사용자의 질문에 정확하게 답변합니다.

## 답변 규칙
1. **출처 기반**: 반드시 제공된 문맥의 정보만 사용합니다.
2. **정확성**: 문맥에 없는 내용은 추측하지 않습니다.
3. **명확성**: 답변은 이해하기 쉽게 구조화합니다.
4. **언어**: 한국어로 답변합니다.

## 답변 불가 시
문맥에서 정보를 찾을 수 없으면:
"제공된 문서에서 해당 정보를 찾을 수 없습니다. 다른 질문을 해주세요."
라고 답변합니다."""),
    ("human", """## 참조 문맥
{context}

## 질문
{question}

## 답변""")
])

# 문서 포메터 작성  : 
def format_docs(docs):
    '''검색된 문서들을 하나의 문자열로 포맷팅'''
    return '\n\n---\n\n'.join([ doc.page_content for doc in docs])

def format_docs_with_source(docs):
    '''출저 정보를 포함하여 문서 포멧팅'''
    formatted = []
    for i ,doc in enumerate(docs, 1):
        source = doc.metadata.get('source','unknown')
        formatted.append(f'문서 {i}: {source}\n{doc.page_content}')
    return '\n\n---\n\n'.join(formatted)

# 테스트
test_docs = retriever.invoke('RAG란 무엇인가요?')
print('검색된 문서 포멧팅 예시')
print(format_docs_with_source(test_docs[:2]))

# RAG 체인 구성
# 기본 RAG 체인(LCEL 사용)
rag_chain = (
    {'context': retriever | format_docs, 'question':RunnablePassthrough()}
    | basic_prompt
    | llm
    | StrOutputParser()
)
print('기본 RAG 체인 구성 완료')
# 출처 포함 RAG 체인
rag_chain_with_source =  (
    {'context': retriever | format_docs_with_source, 'quetion':RunnablePassthrough()}
    | basic_prompt
    | llm
    | StrOutputParser()
)
print('출처 포함 RAG 체인 구성 완료')
'''
체인구조
 질문 ->    retriever           --> 관련 문서 검색
            format_docs         --> 문자열로 변환

            prompt              --> context + question 결합

            llm                 --> 답변 생성

            Strparser           --> 문자열 출력
'''

print('RAG 체인 테스트')
test_questions = [
    "RAG란 무엇이고 어떤 장점이 있나요?",
    "LangChain의 주요 구성 요소를 설명해주세요.",    
    'VectorDB의 종류를 알려주세요'
]

for i , question in enumerate(test_questions,1):
    print(f'테스트 {i} : {question}')
    print('-'*60)
    star_time = time.time()
    # RAG 체인 실행
    asnwer = rag_chain.invoke(question)
    elapsed = time.time() - star_time
    print(f'답변 : {asnwer}')
    # 참조문서
    retrieved_docs = retriever.invoke(question)
    sources = [doc.metadata.get('source','unknown')  for doc in retrieved_docs]
    print(f'참조 문서 : {sources}')
    print(f'소요된 시간 : {elapsed}')

# 고급 RAG 사용
print('RAG 성능향상을 위한 고급 패턴')

print('query transformaton ')
query_transform_prompt = ChatPromptTemplate.from_template(
    '''다음 질문을 검색에 더 적합한 형태로 변환해주세요.
    키워드 중심으로 명확하게 바꿔주세요

    원본질문:{question}

    변환된 검색어 (한 줄로):'''
)
query_chain = query_transform_prompt | llm | StrOutputParser()

orginal_question = 'RAG가 뭔지 좀 알려주세요'
transformed = query_chain.invoke({'question':orginal_question })
print(f'    원본 : {orginal_question}')
print(f'    변환 : {transformed}')