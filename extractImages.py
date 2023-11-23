from rapidocr_pdf import PDFExtracter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PDFMinerLoader

pdf_extracter = PDFExtracter()

pdf_path = './Kearney2019.pdf'
texts = pdf_extracter(pdf_path, force_ocr=False)
print(texts[6])

loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
print(pages[6])

# loader = PDFMinerLoader(pdf_path)
# data = loader.load_and_split()
# print(data[6])