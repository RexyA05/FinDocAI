import io
import requests
from pypdf import PdfReader


HEADERS={
    "User-Agent": "FinDocAI project ritamvision@gmail.com"
}
def download_sec_pdf(pdf_url: str)-> bytes:
    print(f"Downloading PDF from {pdf_url}")
    response=requests.get(pdf_url,headers=HEADERS, timeout=30)
    response.raise_for_status()
    print("Download Successful!")
    return response.content
def extract_text_from_pdf_bytes(pdf_bytes: bytes)->str:
    pdf_file=io.BytesIO(pdf_bytes)
    reader=PdfReader(pdf_file)
    extracted_pages=[]
    total_pages=len(reader.pages)
    print(f"Processing {total_pages} pages...")
    for index,page in enumerate(reader.pages):
        text=page.extract_text()
        if text:
            extracted_pages.append(text)
    full_document_text ="\n\n".join(extracted_pages)
    print(f"Text extraction complete! Total characters extracted: {len(full_document_text)}")
    return full_document_text

if __name__=="__main__":
    sample_pdf_url="https://www.sec.gov/files/form10-k.pdf"
    try:
        raw_pdf_bytes=download_sec_pdf(sample_pdf_url)
        extracted_text=extract_text_from_pdf_bytes(raw_pdf_bytes)
        print("\n---Document Preview (First 500 characters) --")
        print(extracted_text[:500])
        print("-------------------------------")
        print("\nStage 1 Data extraction compelte!")
    except requests.exceptions.RequestException as err:
        print(f"HTTP Error fetching document: {err}")
    except Exception as err:
        print(f"Error parsing document: {err}")