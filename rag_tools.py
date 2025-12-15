import os
from dotenv import load_dotenv
import re
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from qdrant_client.http import models as qmodels


# =========================
# 1. LOAD CONFIG
# =========================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "resume_chunks"

# =========================
# 2. INIT CLIENTS
# =========================

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# =========================
# 3. RAG TOOL FUNCTION
# =========================

def make_snippet(text: str, max_len: int = 400) -> str:
    if not text:
        return ""
    text = text.strip()

    # kalau teks lebih pendek dari max_len, langsung balikin
    if len(text) <= max_len:
        return text

    # potong di batas max_len, tapi mundur sampai spasi terdekat biar gak putus kata
    cut = text[:max_len]
    last_space = cut.rfind(" ")
    if last_space != -1:
        cut = cut[:last_space]
    return cut + "..."

@tool("search_resumes")
def search_resumes(query: str, top_k: int = 5) -> str:
    """
    Search for relevant resume chunks from Qdrant vector database.

    Behavior:
    - If the query contains one or more resume IDs (e.g. 57667857),
      we fetch chunks for those specific IDs so follow-up questions like
      "tell me more about Resume ID: 57667857" or
      "compare 57667857 and 11847784" work reliably.
    - Otherwise, we use normal semantic search.

    The returned text is structured so the RAG LLM can
    describe / compare candidates side by side.
    """

    # 1. Embed query (still useful even for ID mode so Qdrant has a vector)
    query_vector = embeddings.embed_query(query)

    # 2. Detect resume IDs in the query (numbers with 5+ digits)
    raw_ids = re.findall(r"\b\d{5,}\b", query)
    resume_ids = []
    for rid in raw_ids:
        try:
            resume_ids.append(int(rid))
        except ValueError:
            continue

    # Remove duplicates but keep order
    seen = set()
    resume_ids = [rid for rid in resume_ids if not (rid in seen or seen.add(rid))]

    # ==============
    # A. ID MODE
    # ==============
    if resume_ids:
        formatted_sections: list[str] = []
        header_ids = ", ".join(str(rid) for rid in resume_ids)
        formatted_sections.append(
            f"Comparison context for Resume IDs: {header_ids}\n"
            f"(Each section below describes one resume.)"
        )

        # How many chunks per resume to fetch
        per_resume_k = max(1, top_k)

        for rid in resume_ids:
            result = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="resume_id",
                            match=qmodels.MatchValue(value=rid),
                        )
                    ]
                ),
                limit=per_resume_k,
                with_payload=True,
            )

            points = result.points

            if not points:
                formatted_sections.append(
                    f"\n=== Resume ID {rid} ===\n"
                    f"No data found for this resume ID in the database."
                )
                continue

            # Gather snippets for this resume
            snippets = []
            category = None
            for p in points:
                payload = p.payload or {}
                if category is None:
                    category = payload.get("category", "Unknown")

                full_text = payload.get("chunk_text", "")
                snippet = make_snippet(full_text, max_len=350)
                if snippet and snippet not in snippets:
                    snippets.append(snippet)

            # Build section text for this resume
            section_lines = [
                f"\n=== Resume ID {rid} ===",
                f"Category: {category}",
                "Key snippets (work experience / skills / summary):",
            ]
            for s in snippets:
                section_lines.append(f"- {s}")

            formatted_sections.append("\n".join(section_lines))

        return "\n".join(formatted_sections)

    # ==============
    # B. NORMAL SEMANTIC SEARCH MODE
    # ==============
    result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    points = result.points

    if not points:
        return "❌ No relevant resume data found."

    # Format semantic search results (your old behavior)
    formatted_results = []
    for i, p in enumerate(points, start=1):
        payload = p.payload or {}
        full_text = payload.get("chunk_text", "")
        snippet = make_snippet(full_text, max_len=400)

        formatted_results.append(
            f"""
Result {i}
Resume ID: {payload.get('resume_id')}
Category: {payload.get('category')}
Snippet: {snippet}
"""
        )

    return "\n".join(formatted_results)
