from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
documents = loader.load()  # 페이지별 Document 리스트
for doc in documents:
    print(f"소스 : {doc.metadata['source']}")
    print(f"페이지 : {doc.metadata['page']}")
    print(f"컨텐츠 길이 : {len(doc.page_content)}문자")
    print(f"컨텐츠 미리보기 : {doc.page_content[:200]}...")