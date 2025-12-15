import os
import math
import pandas as pd
from dotenv import load_dotenv
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from langchain_openai import OpenAIEmbeddings

# ========= 1. CONFIG =========

# Path ke CSV chunks yang tadi kamu buat
CSV_PATH = "cleaned_resume_chunks.csv"

# Nama collection di Qdrant (bebas, tapi konsisten)
COLLECTION_NAME = "resume_chunks"

# Ukuran vektor untuk model text-embedding-3-small
EMBEDDING_DIM = 1536   # penting: harus cocok dengan model


# ========= 2. LOAD ENV & INSTANTIATE CLIENTS =========

load_dotenv()  # akan baca .env di folder project

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY or not OPENAI_API_KEY:
    raise ValueError("❌ QDRANT_URL / QDRANT_API_KEY / OPENAI_API_KEY belum ter-set di .env")

# Qdrant client (untuk Qdrant Cloud pakai url + api_key)
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,  # ⬅️ increase from default (try 60s first)
)

# OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)


# ========= 3. LOAD CSV =========

df = pd.read_csv(CSV_PATH)

# Pastikan kolomnya sesuai
expected_cols = {"resume_id", "category", "chunk_index", "chunk_text"}
if not expected_cols.issubset(set(df.columns)):
    raise ValueError(f"❌ Kolom CSV tidak lengkap. Ditemukan: {df.columns}")


# ========= 4. CREATE / RECREATE COLLECTION =========

# Kalau kamu mau bersihin koleksi lama setiap run, pakai recreate_collection
# hati-hati: ini akan menghapus data lama
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=qmodels.VectorParams(
        size=EMBEDDING_DIM,
        distance=qmodels.Distance.COSINE,
    ),
)

print(f"✅ Collection '{COLLECTION_NAME}' siap dipakai di Qdrant.")

# ========= 4.1 CREATE PAYLOAD INDEXES =========
# REQUIRED so we can filter by resume_id later

qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="resume_id",
    field_schema=qmodels.PayloadSchemaType.INTEGER,
)

# (Optional, but good for future filters)
qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=qmodels.PayloadSchemaType.KEYWORD,
)

qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="chunk_index",
    field_schema=qmodels.PayloadSchemaType.INTEGER,
)

print("✅ Payload indexes created (resume_id, category, chunk_index)")


# ========= 5. BATCH EMBEDDING & UPSERT =========

BATCH_SIZE = 32  # boleh diubah

num_rows = len(df)
num_batches = math.ceil(num_rows / BATCH_SIZE)
print(f"➡️ Total chunks: {num_rows}, akan diproses dalam {num_batches} batch.")

for batch_idx in range(num_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, num_rows)
    batch = df.iloc[start:end]

    texts = batch["chunk_text"].tolist()

    # 5.1. Buat embedding
    vectors = embeddings.embed_documents(texts)  # list[list[float]]

    # 5.2. Siapkan points untuk Qdrant
    points = []

    # THIS IS THE FIXED PART - properly indented inside the batch loop
    for (row_idx, row), vector in zip(batch.iterrows(), vectors):
        # row_idx adalah index dari DataFrame, bisa kita pakai sebagai integer ID
        point_id = int(row_idx)

        payload = {
            "resume_id": int(row["resume_id"]),
            "category": row["category"],
            "chunk_index": int(row["chunk_index"]),
            "chunk_text": row["chunk_text"],  # simpan full chunk
        }

        points.append(
            qmodels.PointStruct(
                id=point_id,        # sekarang integer, bukan string
                vector=vector,
                payload=payload,
            )
        )

    # 5.3. Upsert ke Qdrant
    MAX_RETRIES = 5

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise  # give up after max retries
            wait_s = 2 ** attempt  # 2,4,8,16...
            print(f"⚠️ Upsert failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            print(f"   ⏳ retrying in {wait_s}s...")
            time.sleep(wait_s)


print("🎉 Selesai! Semua chunks sudah di-upload ke Qdrant.")