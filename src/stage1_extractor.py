import io
import hashlib
import requests
from pathlib import Path
from pypdf import PdfReader
import pdfplumber


HEADERS={
    "User-Agent": "FinDocAI project ritamvision@gmail.com"
}
DATA_DIR=Path("data")
#Downloads a financial PDF document from SEC EDGAR into memory as raw bytes.
def get_document_id(pdf_url: str)->str:
    return hashlib.md5(pdf_url.encode("utf-8")).hexdigest()[:10]


def download_sec_pdf(pdf_url: str)-> bytes:
    print(f"Downloading PDF from {pdf_url}")
    response=requests.get(pdf_url,headers=HEADERS, timeout=30)
    response.raise_for_status()
    print("Download Successful!")
    return response.content


#Parses PDF bytes page by page using pypdf and extracts clean text string.
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    pdf_file = io.BytesIO(pdf_bytes)
    extracted_pages = []
    with pdfplumber.open(pdf_file) as pdf:
        total_pages = len(pdf.pages)
        print(f"Processing {total_pages} pages...")
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_pages.append(text)
    full_document_text = "\n\n".join(extracted_pages)
    print(f"Text extraction complete! Total characters extracted: {len(full_document_text)}")
    return full_document_text

#Fetches and extracts text from a PDF URL with deterministic hash caching: Automatically derives cache filename from MD5 hash of the URL if not provided. Prevents stale cache collisions when switching between different SEC filings.
def get_extracted_text(
        pdf_url: str="https://www.sec.gov/files/form10-k.pdf",
        cache_path: Path | None = None,
        force_download: bool = False)->str:
        DATA_DIR.mkdir(parents=True,exist_ok=True)
        if cache_path is None:
            doc_id=get_document_id(pdf_url)
            cache_path=DATA_DIR / f"extracted_{doc_id}.txt"
        if not force_download and cache_path.exists():
            print(f"[Cache hit] Loading text from cache: {cache_path}")
            return cache_path.read_text(encoding="utf-8")
        print(f"[Cache Miss] Downloading and extracting: {pdf_url}")
        raw_pdf_bytes=download_sec_pdf(pdf_url)
        text=extract_text_from_pdf_bytes(raw_pdf_bytes)
        cache_path.write_text(text, encoding="utf-8")
        print(f"Saved extracted text to cache: {cache_path}")
        return text

if __name__=="__main__":
    sample_pdf_url="https://www.sec.gov/files/form10-k.pdf"
    try:
        extracted_text=get_extracted_text(sample_pdf_url)
        #Displaying a quick preview of the extracted documents
        print("\n---Document Preview (First 500 characters) --")
        print(extracted_text[:500])
        print("-------------------------------")
        print("\nStage 1 Data extraction compelte!")
    except requests.exceptions.RequestException as err:
        print(f"HTTP Error fetching document: {err}")
    except Exception as err:
        print(f"Error parsing document: {err}")