# 1: PDF 문서 로딩 (Docling) 이론

## 목차

1. [PDF 처리의 중요성](#1-pdf-처리의-중요성)
2. [PDF 로더 비교](#2-pdf-로더-비교)
3. [Docling 상세](#3-docling-상세)
4. [표와 이미지 처리](#4-표와-이미지-처리)
5. [한국어 PDF 처리](#5-한국어-pdf-처리)
6. [적용 가이드](#6-적용-가이드)

---

## 1. PDF 처리의 중요성

### 1.1 기업 문서의 현실

```text
기업 내 문서 형식 분포 (추정):

PDF ██████████████████████ 45%
DOCX █████████████ 25%
PPT ██████ 12%
기타 █████████ 18%
```

**PDF가 RAG에서 중요한 이유:**

- 공식 문서, 보고서의 표준 형식
- 레이아웃과 서식이 보존됨
- 표, 이미지, 차트 포함 가능
- 기업 지식의 상당 부분이 PDF로 존재

### 1.2 PDF 처리의 어려움

| 문제 | 설명 | 영향 |
|------|------|------|
| **레이아웃 복잡성** | 다단, 헤더/푸터, 각주 | 텍스트 순서 혼란 |
| **표 구조** | 셀 경계 인식 어려움 | 데이터 손실 |
| **스캔 문서** | 이미지 기반 PDF | OCR 필요 |
| **임베딩 폰트** | 한글 폰트 문제 | 인코딩 오류 |

### 1.3 PDF 처리 파이프라인

```text
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   PDF    │ →  │  파싱    │ →  │  구조화   │ →  │  청킹    │
│  파일    │    │ (Loader) │    │ (마크다운)│    │(Splitter)│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

## 2. PDF 로더 비교

### 2.1 주요 PDF 로더

| 로더 | 장점 | 단점 | 추천 용도 |
|------|------|------|-----------|
| **PyPDFLoader** | 간단, 빠름 | 표 처리 약함 | 단순 텍스트 PDF |
| **PDFPlumberLoader** | 표 추출 우수 | 속도 느림 | 표 많은 문서 |
| **UnstructuredPDFLoader** | 범용, 다기능 | 설정 복잡 | 다양한 형식 |
| **Docling** | 레이아웃 분석 | 리소스 많이 사용 | 복잡한 문서 |

### 2.2 PyPDFLoader (기본)

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
documents = loader.load()  # 페이지별 Document 리스트

# 특징:
# - 가장 빠름
# - 페이지 단위로 분할
# - 메타데이터: page, source
```

**출력 예시:**

```python
Document(
    page_content="이것은 PDF의 텍스트입니다...",
    metadata={"source": "document.pdf", "page": 0}
)
```

### 2.3 PDFPlumberLoader (표 추출)

```python
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("document.pdf")
documents = loader.load()

# 특징:
# - 표 구조 인식
# - 텍스트 좌표 추출 가능
# - PyPDFLoader보다 느림
```

### 2.4 UnstructuredPDFLoader (고급)

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader(
    "document.pdf",
    mode="elements",  # 요소별 분리
    strategy="hi_res"  # 고해상도 분석
)
documents = loader.load()

# 특징:
# - 제목, 본문, 표 등 요소 구분
# - OCR 지원
# - 많은 의존성 필요
```

---

## 3. Docling 상세

### 3.1 Docling이란?

**Docling**은 IBM에서 개발한 오픈소스 문서 변환 라이브러리입니다.

```text
Docling의 강점:
├── 레이아웃 분석 (Layout Analysis)
├── 표 구조 인식 (Table Structure Recognition)
├── 마크다운 변환 (Markdown Export)
├── 다국어 지원 (한국어 포함)
└── OCR 통합 (Tesseract, EasyOCR)
```

### 3.2 설치 방법

```bash
# 기본 설치
pip install docling

# OCR 지원 포함
pip install docling[ocr]

# 전체 기능
pip install docling[all]
```

### 3.3 기본 사용법

```python
from docling.document_converter import DocumentConverter

# 변환기 생성
converter = DocumentConverter()

# PDF 변환
result = converter.convert("document.pdf")

# 마크다운으로 내보내기
markdown_text = result.document.export_to_markdown()

# 텍스트만 추출
plain_text = result.document.export_to_text()
```

### 3.4 Docling의 문서 구조 인식

```text
PDF 입력:
┌────────────────────────────┐
│ [제목] RAG 시스템 개요      │
│                            │
│ 본문 텍스트가 여기에...      │
│                            │
│ ┌──────┬──────┬──────┐    │
│ │ 항목  │ 값1  │ 값2  │    │  ← 표
│ ├──────┼──────┼──────┤    │
│ │ A    │ 100  │ 200  │    │
│ └──────┴──────┴──────┘    │
└────────────────────────────┘

Docling 출력 (마크다운):
# RAG 시스템 개요

본문 텍스트가 여기에...

| 항목 | 값1 | 값2 |
|------|-----|-----|
| A    | 100 | 200 |
```

### 3.5 Docling 파이프라인 옵션

```python
from docling.datamodel.pipeline_options import PdfPipelineOptions

# 파이프라인 옵션 설정
options = PdfPipelineOptions()
options.do_ocr = True           # OCR 활성화
options.do_table_structure = True  # 표 구조 분석
options.table_structure_options.do_cell_matching = True

# 옵션 적용
converter = DocumentConverter(pipeline_options=options)
```

---

## 4. 표와 이미지 처리

### 4.1 표(Table) 처리의 중요성

```text
표가 포함된 PDF 예시:

"2024년 매출 현황은 다음과 같습니다.

| 분기 | 매출액 | 성장률 |
|------|--------|--------|
| Q1   | 100억  | 10%    |
| Q2   | 120억  | 20%    |
"

질문: "2024년 Q2 매출은?"
→ 표 구조를 인식해야 정확한 답변 가능
```

### 4.2 표 추출 방법

**방법 1: PDFPlumber**

```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            # table은 2D 리스트
            for row in table:
                print(row)
```

**방법 2: Docling (권장)**

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")

# 표가 마크다운 형식으로 자동 변환됨
markdown = result.document.export_to_markdown()
```

### 4.3 이미지 처리

```python
# Docling으로 이미지 추출
result = converter.convert("document.pdf")

# 이미지 정보 접근
for item in result.document.iterate_items():
    if item.type == "image":
        print(f"이미지 발견: {item.image_path}")
```

### 4.4 수식 처리

```text
PDF의 수학 수식:
    E = mc²

Docling 출력:
    $E = mc^2$  (LaTeX 형식)
```

---

## 5. 한국어 PDF 처리

### 5.1 한국어 PDF의 특수성

| 문제 | 원인 | 해결책 |
|------|------|--------|
| **깨진 글자** | 폰트 임베딩 문제 | 올바른 인코딩 설정 |
| **띄어쓰기 오류** | 한글 특성 | 후처리 필요 |
| **조사 분리** | 토큰화 문제 | 한국어 토크나이저 |

### 5.2 인코딩 처리

```python
# PyPDFLoader에서 인코딩 문제 해결
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("korean_document.pdf")
documents = loader.load()

# 후처리: 깨진 문자 정리
def clean_korean_text(text):
    import re
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 한글 자모 정리 (필요시)
    return text.strip()

for doc in documents:
    doc.page_content = clean_korean_text(doc.page_content)
```

### 5.3 OCR을 통한 스캔 문서 처리

```python
# Tesseract OCR (한국어)
import pytesseract
from pdf2image import convert_from_path

# PDF를 이미지로 변환
images = convert_from_path("scanned_korean.pdf")

# OCR 실행 (한국어)
texts = []
for image in images:
    text = pytesseract.image_to_string(image, lang='kor')
    texts.append(text)
```

### 5.4 한국어 최적화 청킹

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 한국어 문장 종결을 고려한 분할
korean_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=[
        "\n\n",      # 문단
        "\n",        # 줄바꿈
        "다.",       # 한국어 종결어미
        "요.",
        "습니다.",
        ". ",        # 영어 문장
        " ",         # 공백
        ""           # 문자
    ]
)
```

---

## 6. 적용 가이드

### 6.1 PDF 로더 선택 가이드

```text
PDF 특성 분석:
│
├── 텍스트 위주?
│   └── YES → PyPDFLoader (빠름)
│
├── 표가 많음?
│   └── YES → PDFPlumberLoader 또는 Docling
│
├── 복잡한 레이아웃?
│   └── YES → Docling (레이아웃 분석)
│
├── 스캔 문서?
│   └── YES → Docling (OCR) 또는 Tesseract
│
└── 다국어/한국어?
    └── YES → Docling (다국어 지원)
```

### 6.2 PDF 처리 파이프라인 예시

```python
from pathlib import Path

class PDFProcessor:
    def __init__(self, method="docling"):
        self.method = method
    
    def process(self, pdf_path: str) -> list:
        """PDF를 Document 리스트로 변환"""
        
        if self.method == "pypdf":
            return self._process_pypdf(pdf_path)
        elif self.method == "docling":
            return self._process_docling(pdf_path)
    
    def _process_pypdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    
    def _process_docling(self, pdf_path):
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown = result.document.export_to_markdown()
        
        # 마크다운을 Document로 변환
        return [Document(
            page_content=markdown,
            metadata={"source": pdf_path, "type": "docling"}
        )]
```

### 6.3 대용량 PDF 처리

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

async def process_pdfs_parallel(pdf_paths: list):
    """여러 PDF를 병렬 처리"""
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, process_single_pdf, path)
            for path in pdf_paths
        ]
        results = await asyncio.gather(*tasks)
    
    return results
```

### 6.4 품질 검증

```python
def validate_pdf_extraction(original_pdf, extracted_docs):
    """PDF 추출 품질 검증"""
    
    checks = {
        "text_length": len(extracted_docs) > 0,
        "no_empty_pages": all(len(d.page_content) > 10 for d in extracted_docs),
        "metadata_present": all("source" in d.metadata for d in extracted_docs),
    }
    
    return all(checks.values()), checks
```

---

## 핵심 요약

### 체크리스트

- [ ] PDF 특성에 맞는 로더 선택
- [ ] 표 구조 보존 여부 확인
- [ ] 한국어 인코딩 처리
- [ ] 청킹 전략 최적화
- [ ] 추출 품질 검증

### 포인트

1. **PyPDFLoader**: 빠르고 단순, 표 처리 약함
2. **Docling**: 복잡한 문서에 강함, 마크다운 출력
3. **한국어 PDF**: 인코딩과 띄어쓰기 주의
4. **표 처리**: RAG 품질에 큰 영향