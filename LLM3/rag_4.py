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
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    separators= ['\n\n','\n','.',' ','']
)
# 스플릿 = 청킹
doc_splits =  text_splitter.split_documents(document)
print(f'청킹개수 : {len(doc_splits)}')

# 임베딩 벡터
embedding_model =  OpenAIEmbeddings(model = 'text-embedding-3-small')

vectorstore =  Chroma.from_documents(
    documents=doc_splits,
    collection_name='basic_rag_collection',
    embedding=embedding_model
)

retriever =  vectorstore.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k' : 3}
)

prompt_template =  ChatPromptTemplate.from_messages([
    ('system', 'Answer question based on the given context in Kiorean.'),
    ('human','Context:\n{context}\n\nQuestion:{question}\n\nAnswer:')
])

def format_docs(docs):
    return '\n\n---\n\n'.join([ doc.page_content for doc in docs ])

# LCEL 방식  Runnable객체  실행 invoke  -> 파이프라인
llm = ChatOpenAI(model = 'gpt-4o-mini',temperature=0)
rag_chain = (
    {"context" : retriever | format_docs, 'question':RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
test_question = [
    'RAG란 무엇인가요',
    'LangGraph의 핵심 개념을 설명해주세요',
    '프롬프트 엔지니어링 기법에는 어떤 것들이 있나요?'
]

def ask_question(question):
    '''질문에 대한 답변생성'''
    answer = rag_chain.invoke(question)
    retrieved_docs =  retriever.invoke(question)
    sources =  [ os.path.basename(doc.metadata.get('source', 'unknown')) for doc in retrieved_docs ]
    return answer, sources

# 각 질문에 대한 답변 생성
for i, question in enumerate(test_question, 1):
    print(f'question_{i} : {question}')
    answer, sources = ask_question(question)
    print(f'answer : {answer}')
    print(f'sources : {sources}')