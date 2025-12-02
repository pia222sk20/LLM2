from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    EasyOcrOptions,
    RapidOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
data_folder = Path(__file__).parent / "pdf_doc01.pdf"
print(data_folder)
input_doc_path = data_folder

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True
# Any of the OCR options can be used: EasyOcrOptions, TesseractOcrOptions,
# TesseractCliOcrOptions, OcrMacOptions (macOS only), RapidOcrOptions
# ocr_options = EasyOcrOptions(force_full_page_ocr=True)
# ocr_options = TesseractOcrOptions(force_full_page_ocr=True)
# ocr_options = OcrMacOptions(force_full_page_ocr=True)
ocr_options = RapidOcrOptions(force_full_page_ocr=True)
# ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
pipeline_options.ocr_options = ocr_options

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

doc = converter.convert(input_doc_path).document
md = doc.export_to_markdown()
print(md)