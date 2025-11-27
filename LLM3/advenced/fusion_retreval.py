# 백터검색 BM25 키워드 검색을 RRF 알고리즘으로 결합
# pip install rank_bm25

from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser

class FusionRetrieval:
    '''Fusion Retrieval 기법을 구현한 클래스'''
    def __init__(self, documents:List[Document], retriever_k: int = 5):
        '''
        Args:
            documents : 전체문서 리스트
            retriever_k: 각 검색 방식당 반환할 문서 개수
        '''
        self.documents=documents
        self.retriever_k=retriever_k
        #BM25 Retriever 초기화
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = retriever_k
        print('Fusion Retrieval 초기화 완료')
    def fusion_retrieval(self, question:str, vector_retrieval) -> List[Document]:
        '''
        백터 검색과 BM25 검색 결과를 RRF 알고리즘으로 결합
        Args:
            question : 사용자 질문
            vector_retrieval : 백터검색 retriever 객체
        Returns:
            문서리스트
        '''
        # 백터 검색
        vector_docs = vector_retrieval.invoke(question)
        # BM25 검색
        bm25_docs = self.bm25_retriever.invoke(question)
        print('fusion_retrieval...')
        print(f'    백터검색 : {len(vector_docs)}개 문서')
        print(f'    BM25검색 : {len(bm25_docs)}개 문서')
        # RRF (Reciprocal Rank Fusion) 점수 계산
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
        # 문서객체 반환
        result = []
        for doc_text, score in  sorted_docs[:self.retriever_k]:
            for doc in self.documents:
                if doc.page_content.startswith(doc_text):
                    result.append(doc)
                    break
        print(f'    RRF 통합결과 : {len(result)}개 문서')
        print(f'    RRF는 각 검색 방식의 순위 역수를 합산해서 최종 순서를 결정')        
        return result
    def get_detail_fusion_info(self,question:str, vector_retrieval) -> dict:
        '''
        상세한 Fusion Retreval 정보 반환
        Args:
            question : 사용자 질문
            vector_retrieval : 백터 검색 vector_retrieval 객체
        Returns:
            Fusion Retreval 상세 정보 Dict
        '''
        vector_doc = vector_retrieval.invoke(question)
        bm25_docs = self.bm25_retriever.invoke(question)
        return {
            'vector_search_count' : len(vector_doc),
            'bm25_search_count' : len(bm25_docs),
            'vector_docs' : vector_doc,
            'bm25_docs' : bm25_docs
        }   

if __name__ == '__main__':
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import os
    import warnings    
    from dotenv import load_dotenv
    # 경고 메세지 삭제
    warnings.filterwarnings('ignore')
    load_dotenv()
    # openapi key 확인
    api_key = os.getenv('OPENAI_API_KEY')
    print('문서 준비')
    loader = TextLoader('document.txt', encoding='utf-8')
    documents = loader.load()
    print('청크생성')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 70
    )
    chunks = splitter.split_documents(documents)
    print(f'{len(chunks)}개 청크 생성')
    print(f'백터스토어 준비')
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    import os
    embeddings =  OpenAIEmbeddings(model = 'text-embedding-3-small')
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory = './chroma_db'
    )
    print(f'백터스토어 준비완료')
    print('리트리버 생성')
    retrieval = vectorstore.as_retriever(search_kwargs = {'k':3})
    
    question = 'LangChain의 요소는 무엇인가요?'
    print('Fusion Retrieval사용')
    fusion = FusionRetrieval(chunks,3)
    fusion_docs = fusion.fusion_retrieval(question,retrieval)
    print(f'fusion 검색 결과 개수 : {len(fusion_docs)}개')
    print('검색 결과.....')    
    for fusion_doc in fusion_docs:
        print(fusion_doc.page_content)