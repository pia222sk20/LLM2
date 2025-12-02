# from langchain_community.document_loaders import PDFPlumberLoader
# loader = PDFPlumberLoader("document_table.pdf")
# documents = loader.load()
# for doc in documents:
#     print(f"소스 : {doc.metadata['source']}")
#     print(f"페이지 : {doc.metadata['page']}")
#     print(f"컨텐츠 길이 : {len(doc.page_content)}문자")
#     print(f"컨텐츠 미리보기 : {doc.page_content[:200]}...")

from pypdf import PdfWriter, PdfReader
import pdfplumber
PDF_PATH = r'C:\2.Lecture\LLM2\LLM3\PDF\document_table.pdf'


# 표를 마크다운 형식으로 반환
def table_to_markdown(table:list) -> str:
    '''표를 마크다운 형식으로 반환'''
    if not table:
        return ""
    lines = []
    # 헤더
    header = table[0]
    lines.append("|" + "|".join( str(cell) for cell in header ) + "|"   )
    lines.append("|" + "|".join(["---"] * len(header)) + "|"  )

    # 데이터행
    for row in table[1:]:
        lines.append("|" + "|".join( str(cell)  for cell in row ) + "|")
    
    return '\n'.join(lines)


with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
        # 텍스트 추출
        text  = page.extract_text()
        # 표 추출
        tables = page.extract_tables()
        for table in tables:
            print(table_to_markdown( [row for row in table] ))
                


