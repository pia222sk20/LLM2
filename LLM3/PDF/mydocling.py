import os
import warnings
warnings.filterwarnings('ignore')

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Literal

# powershell 에서 

# setx HF_HUB_DISABLE_SYMLINKS 1

# 터키널 종료하고 재 시작(VScode)


# 공식문서의 내용  https://github.com/docling-project/docling
# source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
# converter = DocumentConverter()
# result = converter.convert(source)
# print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"

# # Docling 변환기
# converter = DocumentConverter()
# # pdf -> Docling Document 변환
# file_path = r'C:\2.Lecture\LLM2\LLM3\PDF\document_table.pdf'
# result = converter.convert(file_path)
# # markdown 추출(표 구조 보존)
# markdown_content = result.document.export_to_markdown()
# print(markdown_content)

class DoclingPDFLoader:
    '''Docling을 사용한 pdf 로더'''
    def __init__(self,file_path:str):
        self.file_path = file_path
    def load(self) -> List[Document]:
        '''PDF를 로드하고 Document 리스트로 반환'''
        converter = DocumentConverter()
        result = converter.convert(self.file_path)
        markdown_content = result.document.export_to_markdown()
        # langchain의 Document 형식으로 변환
        documents = [
            Document(
                page_content=markdown_content,
                metadata={
                    'source':self.file_path,
                    'loader':'docling',
                    'format':'markdown'
                }
            )
        ]
        return documents
from langchain_community.document_loaders import PyPDFLoader    
class SimplePDFLoader:
    '''기본 pdf 로더
    간단한 텍스트 추출에 적합
    표 구조는 보존되지 않음
    이미지의 텍스트는 잘 안됨
    '''
    def __init__(self, file_path:str):
        self.file_path = file_path
    def load(self) -> List[Document]:
        '''PDF 로더(텍스트만 추출)'''
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        # 메타데이터에 로더 정보 추가
        for doc in documents:
            doc.metadata['loader'] = 'pypdf'
        return documents

