# 4. Contextual Compressioin (문맥 압축) - 관련 부분만 추출
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
    loader_kwargs = {'encoding':'utf-8'},
    # recursive=True
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

# 문맥압축 프롬프트


question = 'VectorDB의 종류를 알려주세요'

# 1. 문맥압축 프롬프트를 실행
compress_prompt = ChatPromptTemplate.from_template(
'''
다음 문서에서 질문과 관련된 부분만 추출하세요.
관련 없는 부분은 제외하고, 관련 있는 내용만 그대로 출력하세요.
관련 내용이 없으면 "관련 없음"이라고 출력하세요.

문서: {document}
질문: {question}

관련 내용:
'''
)

docs = retriever.invoke(question)
compressed = []
sources = []
for doc in docs:
    document = doc.page_content
    compress_chain = compress_prompt | llm | StrOutputParser()
    compress_result = compress_chain.invoke({'question' :question, 'document': document })

    if "관련 없음" not in compress_result:
        compressed.append(compress_result) 
        sources.append( os.path.basename(doc.metadata.get('source',"") ))

context = '\n\n---\n\n'.join(compressed)    

# 최종 답변
rag_prompt = ChatPromptTemplate.from_messages([
    ('system','제공된 문맥을 바탕으로 한국어로 답변하세요'),
    ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
])
rag_prompt_chain = rag_prompt | llm | StrOutputParser()
result = rag_prompt_chain.invoke({'context' : context, 'question':question})
print(result, sources)