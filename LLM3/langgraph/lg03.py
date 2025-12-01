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

def langgraph_rag():
    '''VectorDB 기반 LangGraph RAG'''
    # 상태 정의
    class RAGState(TypedDict):
        question:str
        documents : List[Document]
        doc_socres : List[float]
        search_type : str
        answer : str
    # 문서 로드    
    # C:\LLM\LLM3\advenced\sample_docs\langgraph_rag\langchain_intro.txt
    path = 'C:/LLM/LLM3/advenced/sample_docs'
    loader = DirectoryLoader(
        path = path,
        glob = '**/*.txt',
        loader_cls = TextLoader,
        loader_kwargs = {'encoding':'utf-8'},        
    )
    docs = loader.load()

    # VectorDB 구축
    text_spliter =  RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50,separators= ['\n\n','\n','.',' ',''])
    splits = text_spliter.split_documents(docs)

    vectorstores = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model='text-embedding-3-small'),
        collection_name='langgraph'
    )
    print(f'VectorDB 구축 완료 청크개수 : {len(splits)}')
    # llm 초기화
    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)

    # 노드 함수들
        # 리트리버 함수
    def retrieve_node(state:RAGState)->dict:
        '''검색 노드'''
    def grade_documents_node(state:RAGState)->dict:
        '''문서평가 노드'''
    def web_search_node(stae:RAGState)->dict:
        '''웹검색 노드(시뮬레이션)'''
    def generate_node(state:RAGState)->dict:
        '''생성노드'''
    # 조건 함수
    def decide_to_generate(state:RAGState)-> Literal['generate','web_search']:
        '''조건부 분기 함수'''    
    # 그래프 구축(add_node  add_edge  add_conditional_edges)
    graph = StateGraph(RAGState)
    graph.add_node('retriever',retrieve_node)
    graph.add_node('grade',grade_documents_node)
    graph.add_node('web_search',web_search_node)
    graph.add_node('generate',generate_node)

    graph.add_edge(START, 'retriever')
    graph.add_edge('retriever', 'grade')
    graph.add_conditional_edges(
        'grade',
        decide_to_generate,
        { 'generate':'generate', 'web_search': 'web_search'}
    )
    graph.add_edge('web_search', 'generate')
    graph.add_edge('generate', END)
    # 그래프 컴파일
    app = graph.compile()
    # 그래프 invoke(질문)
    test_qeustion = [
        'LangGraph의 핵심 개념을 설명해 주세요',
        'RAG란 무엇인가요?',
        '오늘 서울 날씨는 어떤가요?'  # 내부 문서에 없음
    ]






# # 조건부 엣지가 포함된 그래프
# def conditional_graph():
#     '''조건부 엣지가 포함된 LangGraph
#     검색결과에 따라 다른 경로로 분기'''
#     # 상태 정의
#     class ConditionalState(TypedDict):
#         question:str
#         documents : List[Document]
#         search_type : str
#         answer : str
#     # 내부문서  LoadeText or .....
#     INTERNAL_DOCS = {
#             "회사" : [Document(page_content="우리 회사의 AI전략은 RAG 시스템 구축입니다.")],
#             "정책" : [Document(page_content="사내 데이터 보안 정책은 외부 공유 금지입니다.")],
#     }
#     # 노드함수들을 구현
#     def internal_search_node(state:ConditionalState) -> dict:
#         '''내부 문서 검색'''
#         question = state['question']
#         documents = []
#         for keyword,docs in INTERNAL_DOCS.items():
#             if keyword in question:
#                 documents.extend(docs)
#         return {'documents':documents,'search_type':'internal'}
    
#     def web_search_node(state:ConditionalState) -> dict:
#         '''웹 검색(시뮬레이션)'''
#         mock_result = Document(
#             page_content= f"{state['question']}에 대한 웹 검색 결과 입니다.",
#             metadata = {'source':'web'}
#         )
#         return {'documents':[mock_result],'search_type':'web'}
#     def generate_node(state:ConditionalState) -> dict:
#         '''답변 생성'''
#         llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)
#         context = '\n'.join([ doc.page_content for doc in state['documents']  ])
#         prompt = ChatPromptTemplate.from_template(
#             '''컨텍스트:{context}\n\n 질문:{question}\n\n답변:'''

#         )
#         chain = prompt | llm | StrOutputParser()
#         answer = chain.invoke({'context':context, 'question': state['question']})
#         return {'answer' : f"[{state['search_type']}] {answer}"}
#     # 조건 함수
#     def decide_search_type(state:ConditionalState) -> Literal['generate','web_search']:
#         '''검색결과에 따라 분기'''
#         if state['documents']:
#             return 'generate'
#         else:
#             return 'web_search'
    
#     # 그래프 구축
#     graph = StateGraph(ConditionalState)
#     graph.add_node('internal_search', internal_search_node)
#     graph.add_node('web_search', web_search_node)
#     graph.add_node('generate', generate_node)

#     graph.add_edge(START,'internal_search')
#     graph.add_conditional_edges(
#         'internal_search',
#         decide_search_type,
#         {
#             'generate' :'generate',
#             'web_search':'web_search'
#         }
#     )
#     graph.add_edge('web_search','generate')
#     graph.add_edge('generate', END)

#     app = graph.compile()

#     # 테스트 1 : 내부문서에 있는 질문
#     print('\n[테스트1] 내부 문서가 존재하는 경우')
#     result1 = app.invoke({
#         'question' : '회사 AI 전략은?',
#         'documents':[],
#         'search_type':'',
#         'answer':''
#     })
#     print(f"답변 : {result1['answer']}")
#     # 테스트 2 : 내부문서에 없는 질문
#     print('\n[테스트2] 내부 문서가 없는 경우 -> 웹 검색')
#     result2 = app.invoke({
#         'question' : '오늘날씨는??',
#         'documents':[],
#         'search_type':'',
#         'answer':''
#     })
#     print(f"답변 : {result2['answer']}")

# # 조건부 분기 테스트
# conditional_graph()