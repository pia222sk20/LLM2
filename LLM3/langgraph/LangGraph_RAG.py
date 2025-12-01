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
        doc_scores : List[float]
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
        quesiton = state['question']
        docs_with_scores = vectorstores.similarity_search_with_score(quesiton, k = 3)
        documents =  [ doc for doc,score in docs_with_scores]
        scores =  [ 1-score for doc,score in docs_with_scores]

        print(f' [retriever] {len(documents)}개 문서 검색됨')        
        return {'documents': documents, 'doc_scores':scores, 'search_type':'internal'}   # state 업데이트

    def grade_documents_node(state:RAGState)->dict:
        '''문서평가 노드'''
        threshold = 0.3
        filtered_docs, filtered_scores = [],[]
        for doc, score in zip(state['documents'],state['doc_scores']):
            if score >= threshold:
                filtered_docs.append(doc); filtered_scores.append(score)
        print(f"[grade] {len(state['documents'])}개 --> {len(filtered_docs)}개 문서 유지")
        return {'documents' : filtered_docs, 'doc_scores':filtered_scores}

    def web_search_node(state:RAGState)->dict:
        '''웹검색 노드(시뮬레이션)'''
        web_result = Document(
            page_content=f"웹 검색 결과 : {state['question']}에 대한 최신 결과 입니다.",
            metadata = {'source':'web_search'}
        )
        return {'document':[web_result],'doc_scores':[0.8], 'search_type':'web'}

    def generate_node(state:RAGState)->dict:
        '''생성노드'''  
        context = '\n'.join([ doc.page_content for doc in state['documents']])
        prompt = ChatPromptTemplate.from_messages([
            ('system','제공된 문맥을 바탕으로 한국어로 답변하세요'),
            ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
        ])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({'context':context, 'question' : state['question'] })
        return {'answer':answer}
    # 조건 함수
    def decide_to_generate(state:RAGState)-> Literal['generate','web_search']:
        '''조건부 분기 함수'''    
        if state['documents'] and len(state['documents']) > 0:
            return 'generate'
        else:
            return 'web_search'

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
    # 각 질문에 대한 출력
    for question in test_qeustion:        
        result = app.invoke({
            'question':question,
            'documents' : [],
            'doc_scores' : [],
            'search_type' : "",
            'answer' : ""
        })

        print(f'\n 답변 :\n {result['answer']}')
        print(f'\n 검색유형 :{result['search_type']}, 참조문서 : {len(result['documents'])}개')


if __name__ == '__main__':
    langgraph_rag()