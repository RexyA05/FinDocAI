import re
from src.stage1_extractor import get_extracted_text
def preprocess_text(raw_text: str) -> str:
    cleaned_text=re.sub(r'\s+',' ',raw_text)
    return cleaned_text.strip()
def chunk_text_by_words(text:str,chunk_size: int = 500,overlap: int =50)->list[dict]:
    if overlap>=chunk_size:
        raise ValueError("chunk_size must be strictly greater than overlap")
    words=text.split()
    if not words:
        return []
    chunks=[]
    start_idx=0
    chunk_id=0
    step=chunk_size-overlap
    while start_idx < len(words):
        end_idx=min(start_idx+chunk_size,len(words))
        chunk_words=words[start_idx:end_idx]
        chunk_str="".join(chunk_words)
        chunks.append({
            "chunk_id": chunk_id,
            "text":chunk_str,
            "word_count":len(chunk_words),
            "start_word_index":start_idx,
            "end_word_index":end_idx
        })
        chunk_id+=1
        if end_idx==len(words):
            break
        start_idx+=step
    return chunks
if __name__=="__main__":
    sample_pdf_url="https://www.sec.gov/files/form10-k.pdf"
    print("Stage 2: Preprocessing & Chunking Pipeline")
    try:
        raw_text=get_extracted_text(sample_pdf_url)
        cleaned_text=preprocess_text(raw_text)
        print(f"n[Preprocessing] Raw: {len(raw_text)} chars | Cleaned: {len(cleaned_text)} chars")
        chunks=chunk_text_by_words(cleaned_text, chunk_size=500, overlap=50)
        print(f"[Chunking Complete] Generated {len(chunks)} text chunks.\n "+"="*60)
        if len(chunks)>0:
            print(f"PREVIEW: Chunk 0 (Words {chunks[0]['start_word_index']} to {chunks[0]['end_word_index']}) ")
            print(chunks[0]['text'][:250]+"...")
            print(f"Total Words: {chunks[0]['word_count']}")
        if len(chunks)>1:
            print("\n"+"-"*60 +"\n")
            print(f"PREVIEW: Chunk 1 (Words {chunks[1]['start_word_index']} to {chunks[1]['end_word_index']})")
            print(chunks[1]['text'][:250]+"...")
            print(f"Total words: {chunks[1]['word_count']}")
        print("="*60)
        print("\nStage 2 complete!")
    except Exception as err:
        print(f"Pipeline error: {err}")