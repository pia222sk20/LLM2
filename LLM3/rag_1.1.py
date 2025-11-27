# 문서로딩 청킹
# RAG : 외부지식을 활용해서 LLM의 답변 품질을 향상 
# 원본문서(pdf txt doc) -> 문서로딩(loader) ->청킹(Splitter) -< 임베딩/저장(VectorDB)

from langchain_core.documents import Document
doc = Document(
    page_content = '우리회사는 근속 10년이상이면 포상제도가 있습니다.',
    metadata={
        "source" : '010001.pdf',
        'page':1,
        'author' : '작성자',
        'created_at' : "2025-11-27"
    }
)
# 문서로더
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, DirectoryLoader
loader = TextLoader('document.txt',encoding='utf-8')
documents = loader.load()
print(documents)
 
print('='*100)

loader =  WebBaseLoader("https://www.hani.co.kr/arti/science/science_general/1231476.html")
documents = loader.load()
print(documents)

print('='*100)

loader = DirectoryLoader(
    './',
    glob = "**/*.txt",
    loader_cls = lambda path : TextLoader(file_path=path, encoding='utf-8'),
    show_progress=True
)
documents = loader.load()
print(documents)

# 청킹  긴 문서를 작은 조각(청킹)으로 분할
# LLM이 컨텍스트를 제한
# 검색의 정확도 향상(큰문서  관련없는 정보 포함)  작은 청크->질문과 관련된 부분만 검색

# 텍스트 분할기
from langchain_text_splitters import CharacterTextSplitter
spliiter = CharacterTextSplitter(
    separator = '\n',   # 분할 기준
    chunk_size=1000,  # 최대크기
    chunk_overlap=200
)

# 권장  여러 구분자를 계층적으로 시도
from langchain_text_splitters import RecursiveCharacterTextSplitter
spliiter = RecursiveCharacterTextSplitter(
    separators = ['\n','\n\n','.'],   # 분할 기준
    chunk_size=1000,  # 최대크기
    chunk_overlap=200
)