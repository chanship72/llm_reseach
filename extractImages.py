# from rapidocr_pdf import PDFExtracter
# from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders import PDFMinerLoader

# pdf_extracter = PDFExtracter()

# pdf_path = './Kearney2019.pdf'
# texts = pdf_extracter(pdf_path, force_ocr=False)
# print(texts[6])

# loader = PyPDFLoader(pdf_path)
# pages = loader.load_and_split()
# print(pages[6])

# loader = PDFMinerLoader(pdf_path)
# data = loader.load_and_split()
# print(data[6])

# from PyPDF2 import PdfReader
# import fitz  # PyMuPDF
# import io
# from PIL import Image

# # Load the PDF file using PyMuPDF (fitz)
# pdf_path = 'TSLA/TSLA-Q1-2021-Update.pdf'
# pdf_document = fitz.open(pdf_path)

# # Function to extract images from a PDF page
# def extract_images_from_page(page):
#     images = []
#     for img in page.get_images(full=True):
#         xref = img[0]
#         base_image = pdf_document.extract_image(xref)
#         image_bytes = base_image["image"]
#         image = Image.open(io.BytesIO(image_bytes))
#         images.append(image)
#     return images

# # Extracting images from all pages
# extracted_images = []
# for page_number in range(len(pdf_document)):
#     page = pdf_document.load_page(page_number)
#     page_images = extract_images_from_page(page)
#     extracted_images.extend(page_images)

# # Number of images extracted
# num_images_extracted = len(extracted_images)
# num_images_extracted

import camelot

# File path for the PDF
pdf_file_path = 'TSLA/TSLA-Q1-2021-Update.pdf'

# Extract tables from the PDF
tables = camelot.read_pdf(pdf_file_path, flavor='stream', pages='all')

# Counting the number of tables extracted
num_tables = len(tables)
print(tables[0].df)