import pandas as pd
import re
from bs4 import BeautifulSoup

# 1. Load raw CSV
df = pd.read_csv("Resume.csv")

# 2. Cleaning function
def clean_resume_text(text: str) -> str:
    if pd.isna(text):
        return ""
    
    # Remove HTML (just in case)
    text = BeautifulSoup(text, "html.parser").get_text()

    # Lowercase
    text = text.lower()

    # Replace bullets
    text = re.sub(r"(?m)^[\s•*·\-]+", "- ", text)  # versi sedikit lebih aman

    # Remove unwanted characters (keep letters, numbers, basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9\n.,;:!?/()\- ]", " ", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Normalize some patterns
    text = text.replace(" .", ".")
    
    return text.strip()

# 3. Chunking function (berbasis jumlah kata sederhana)
def chunk_text(text: str, max_words: int = 300):
    words = text.split()
    chunks = []
    current = []

    for w in words:
        current.append(w)
        if len(current) >= max_words:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks

# 4. Apply cleaning + chunking to all resumes
rows = []  # akan jadi list of dict → DataFrame chunks

for idx, row in df.iterrows():
    resume_id = row["ID"]
    category = row["Category"]
    raw_text = row["Resume_str"]

    cleaned = clean_resume_text(raw_text)
    
    if not cleaned:
        continue
    
    chunks = chunk_text(cleaned, max_words=300)

    for i, chunk in enumerate(chunks):
        rows.append({
            "resume_id": resume_id,
            "category": category,
            "chunk_index": i,
            "chunk_text": chunk,
            "original_char_length": len(cleaned)
        })

chunks_df = pd.DataFrame(rows)

# 5. Save cleaned full + chunks (optional tapi rapi)
df["Cleaned_Resume"] = df["Resume_str"].apply(clean_resume_text)
df.to_csv("/Users/carolineliem/Documents/cleaned_resume_full.csv", index=False)

chunks_df.to_csv("/Users/carolineliem/Documents/cleaned_resume_chunks.csv", index=False)

print("✅ Saved cleaned_resume_full.csv and cleaned_resume_chunks.csv")
print("   Total resumes:", len(df))
print("   Total chunks:", len(chunks_df))

