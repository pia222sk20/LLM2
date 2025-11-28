# 3. Self-RAG               (자기 보정) - 문서 관련성 평가
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
# self-RAG ( 자기 보정 RAG)
print(f'3. self-RAG')
print(f'검색된 문서의 관련성을 평가하여 필터링합니다.\n')
# 프롬프트
check_prompt = ChatPromptTemplate.from_template("""
다음 문서가 질문에 관련이 있는지 평가하세요
'yes 또는 'no'로만 답변하세요
                                 
문서: {document}
질문: {question}
관련성:""")
# LCEL 체인 구성
check_prompt_chain = check_prompt | llm | StrOutputParser()

def filler_relevant_docs(docs, question):
    '''관련 있는 문서만 필터링'''
    relevant = []
    for doc in docs:
        result = check_prompt_chain.invoke({'document' : doc.page_content , 'question':question})
        is_relevant = 'yes' in result.lower()
        print(f'    -{doc.page_content[:50]}... : {"Relevent" if is_relevant else "Not Relevent"}')
        if is_relevant:
            relevant.append(doc)
    return relevant

# 관련성을 평가후 답변생성

#1. 문서를 검색(리트리버를 이용해서 )
question = '환율이 급격히 상승한 이유는?'
docs = retriever.invoke(question)
print(f'리트리버가 찾은 문서수 : {len(docs)}개')
# 관련성 평가
relevant_docs =  filler_relevant_docs(docs,question)
print(f' relevant_docs 개수 : {len(relevant_docs)}개')

if not relevant_docs:
    raise ValueError('관련있는 문서가 없어서 답변을 종료합니다.다른 질문을 입력하세요')


def format_docs(docs):
    '''문서를 문자열로 포멧팅'''
    return '\n\n---\n\n'.join([ doc.page_content for doc in docs ])

context = format_docs(relevant_docs)
# 답변 생성
# RAG프롬프트
rag_prompt = ChatPromptTemplate.from_messages([
    ('system','제공된 문맥을 바탕으로 한국어로 답변하세요'),
    ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
])
answer_chain = rag_prompt | llm | StrOutputParser()
answer = answer_chain.invoke({'context' : context, 'question' : question})
print(f' \n\n답변 : {answer}')
sources = [ os.path.basename(doc.metadata.get('source',"")) for doc in relevant_docs]
print(f' 근거 : {sources}')
