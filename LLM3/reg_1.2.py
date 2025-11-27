import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import(
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

# 샘플 도큐먼트 객체 생성
example_doc = Document(
    page_content = '이것은 예제 문서의 내용입니다.',
    metadata = {'source':'example.txt', 'page':1,'author':'홍길동'}
)
print('예제 Document 객체')
print(f'page_content : {example_doc.page_content}')
print(f'page_content : {example_doc.metadata}')

#샘플 문서 생성(외부문서 시뮬레이션)
sample_documents = [
    Document(
        page_content="""
        LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션 개발을 위한 프레임워크입니다.
        
        LangChain의 주요 구성 요소:
        1. Models: 다양한 LLM 제공자(OpenAI, Anthropic, Google 등)와 통합
        2. Prompts: 프롬프트 템플릿 관리 및 최적화
        3. Chains: 여러 구성 요소를 연결하는 파이프라인
        4. Memory: 대화 맥락을 유지하기 위한 메모리 시스템
        5. Indexes: 문서 검색을 위한 인덱싱 도구
        6. Agents: 도구를 사용하여 복잡한 작업을 수행하는 에이전트
        
        LangChain Expression Language (LCEL)은 체인을 구성하는 선언적 방식으로,
        파이프(|) 연산자를 사용하여 컴포넌트들을 직관적으로 연결할 수 있습니다.
        """,
        metadata={"source": "langchain_intro.txt", "topic": "framework", "importance": "high"}
    ),
    Document(
        page_content="""
        RAG (Retrieval-Augmented Generation)는 검색 증강 생성 기술입니다.
        
        RAG의 작동 원리:
        1. 사용자 질문을 임베딩 벡터로 변환합니다.
        2. 벡터 데이터베이스에서 유사한 문서를 검색합니다.
        3. 검색된 문서를 컨텍스트로 사용하여 LLM이 답변을 생성합니다.
        
        RAG의 장점:
        - 최신 정보를 반영할 수 있습니다. LLM의 학습 데이터 이후 정보도 활용 가능합니다.
        - 환각(Hallucination)을 감소시킵니다. 실제 문서 기반으로 답변하기 때문입니다.
        - 출처를 명시할 수 있습니다. 어떤 문서에서 정보를 가져왔는지 추적 가능합니다.
        - 도메인 특화가 가능합니다. 특정 분야의 문서만 사용하여 전문적인 답변을 제공합니다.
        
        RAG의 핵심 구성요소: Retriever(검색기), Generator(생성기), VectorStore(벡터저장소)
        """,
        metadata={"source": "rag_concept.txt", "topic": "technique", "importance": "high"}
    ),
    Document(
        page_content="""
        VectorDB(벡터 데이터베이스)는 고차원 벡터를 효율적으로 저장하고 검색하는 데이터베이스입니다.
        
        주요 VectorDB 솔루션:
        - ChromaDB: 로컬 개발에 적합한 오픈소스 솔루션. 파이썬 네이티브로 설치가 간편합니다.
        - Pinecone: 완전 관리형 클라우드 서비스. 대규모 프로덕션 환경에 적합합니다.
        - Weaviate: 그래프 기반 벡터 데이터베이스. 하이브리드 검색을 지원합니다.
        - FAISS: Facebook에서 개발한 고성능 라이브러리. 대용량 벡터 검색에 최적화되어 있습니다.
        - Milvus: 분산 환경을 지원하는 오픈소스 솔루션입니다.
        
        임베딩(Embedding)은 텍스트를 숫자 벡터로 변환하는 과정으로,
        의미적으로 유사한 텍스트는 벡터 공간에서 가까운 위치에 배치됩니다.
        예를 들어, "고양이"와 "강아지"는 "자동차"보다 벡터 공간에서 더 가깝습니다.
        """,
        metadata={"source": "vectordb_intro.txt", "topic": "database", "importance": "medium"}
    ),
]



# 텍스트 분할기 
print('텍스트 분할기')
# 단순 문자기반 분할기
char_splitter = CharacterTextSplitter(
    separator = '\n',
    chunk_size=200,
    chunk_overlap = 30,
    length_function =  len
)
# 첫번째 문서로 테스트
test_doc = sample_documents[0]
char_splits = char_splitter.split_documents([test_doc])
print(f'원본 문서길이 : {len(test_doc.page_content)}자')
print(f'CharacterTextSplitter 결과 : {len(char_splits)}개 청크')
print(f'청크별 미리보기')
for i, chunk in enumerate(char_splits[:3],1):
    preview = chunk.page_content.strip()[:80].replace('\n',' ')
    print(f' 청크{i} ( {len(chunk.page_content)}자: {preview}  )')

# RecursiveCharacterTextSplitter
print(f'RecursiveCharacterTextSplitter 적용')
recursive_splitter =  RecursiveCharacterTextSplitter(
     chunk_size=300,
     chunk_overlap=50,
     separators = ['\n\n','\n','.',' ',''],
     length_function=len
)
# 모든 문서를 청크로 분할
doc_splits = recursive_splitter.split_documents(sample_documents)
print(f'원본 문서 : {len(sample_documents)}개')
print(f'RecursiveCharacterTextSplitter 결과 : {len(doc_splits)}개 청크')


# 청킹 결과 저장 (pickle 사용)
import pickle
# 최종 분할설정(중간크기)
final_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50,
    separators = ['\n\n','\n','.',' ',''],
)
final_chunks = final_splitter.split_documents(sample_documents)
# 파일로저장
output_path = 'chunks_output.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(final_chunks,f)
print(f'저장완료')    
print(f'파일명 : {output_path}')    
print(f'청크수 : {len(final_chunks)}')