from __future__ import annotations

from typing import Literal, List
from typing_extensions import TypedDict, Annotated

from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages

from rag_tools import search_resumes  # tool yang tadi kamu buat
import re

# =========================
# 1. LOAD ENV & CONFIG
# =========================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

# Model untuk agents
# Kamu bisa ganti ke model lain kalau mau (misalnya gpt-4.1-mini, gpt-4o-mini, dll)
SUPERVISOR_MODEL = "gpt-4.1-mini"
RAG_MODEL = "gpt-4.1-mini"
CHAT_MODEL = "gpt-4.1-mini"

from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Initialize Langfuse client (optional, for custom logs)
langfuse = get_client()

# Initialize Langfuse CallbackHandler for LangChain (tracing)
try:
    langfuse_handler = CallbackHandler()
except Exception:
    langfuse_handler = None  # app tetap jalan kalau env Langfuse belum di-set



# Helper function to extract Resume IDs from text
def extract_resume_ids(text: str) -> list[int]:
    """
    Extract Resume IDs from RAG context text.
    Looks for patterns like 'Resume ID: 123456'.
    """
    if not text:
        return []
    ids = re.findall(r"Resume ID:\s*(\d+)", text)
    return [int(x) for x in ids]
# =========================
# 2. STATE DEFINITION
# =========================

class AgentState(TypedDict):
    """
    State that flows through the graph.
    - messages: chat history
    - route: which agent the supervisor chose
    - token_usage: optional LLM usage metadata (for UI display)
    """
    messages: Annotated[List[AnyMessage], add_messages]
    route: str
    token_usage: dict | None
    selected_resume_ids: list[int] | None  # optional, for future use


# =========================
# 3. INIT LLMs
# =========================

common_llm_kwargs = {}
if langfuse_handler is not None:
    common_llm_kwargs["callbacks"] = [langfuse_handler]

supervisor_llm = ChatOpenAI(
    model=SUPERVISOR_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0,
    **common_llm_kwargs,        # ⬅️ Langfuse tracing here
)

rag_llm = ChatOpenAI(
    model=RAG_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.2,
    **common_llm_kwargs,        # ⬅️ Langfuse tracing here
)

chat_llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.7,
    **common_llm_kwargs,        # ⬅️ Langfuse tracing here
)


# =========================
# 4. SUPERVISOR NODE
# =========================

SUPERVISOR_SYSTEM_PROMPT = """
You are the supervisor of a multi-agent system.

Your task:
- If the user asks about resume content, work experience, skills, job categories, candidate recommendations, or anything that requires information from the resume dataset, choose: RAG_AGENT
- If the user only asks general questions, small talk, or topics unrelated to the resume dataset (such as weather, motivation, general theory, etc.), choose: CHAT_AGENT

Return ONLY one of the following:
- RAG_AGENT
- CHAT_AGENT
"""

def supervisor_node(state: AgentState) -> AgentState:
    messages = state["messages"]

    input_messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + messages

    response = supervisor_llm.invoke(input_messages)
    decision = response.content.strip().upper()

    if "RAG" in decision:
        route = "RAG_AGENT"
    else:
        route = "CHAT_AGENT"

    # only update the route, keep other fields in state
    new_state: AgentState = dict(state)
    new_state["route"] = route
    return new_state


def route_from_state(state: AgentState) -> Literal["RAG_AGENT", "CHAT_AGENT"]:
    """
    This function is used by LangGraph to select the next edge.
    """
    route = state.get("route", "CHAT_AGENT")
    if route not in ("RAG_AGENT", "CHAT_AGENT"):
        return "CHAT_AGENT"
    return route


# =========================
# 5. RAG AGENT NODE
# =========================

RAG_SYSTEM_PROMPT = """
You are an assistant that answers questions strictly based on candidate resume data.

You will be given:
- The user's question
- Retrieved resume information from the vector database as context

The context may contain one or more clearly separated sections such as:
"=== Resume ID 123456 === ..."

=====================
STRICT RULES
=====================
- Use ONLY the provided resume context as your source of truth.
- Do NOT use outside knowledge or assumptions.
- Do NOT hallucinate or invent any candidate details.
- If requested information is missing, say exactly:
  "not specified in the context".
- Write in clear, professional, and concise language.
- Output MUST be valid Markdown (no HTML).

=====================
BEHAVIOR GUIDELINES
=====================

1) SINGLE RESUME MODE
If the user asks about ONE resume ID
(e.g. "tell me more about Resume ID 57667857"):

- Produce a focused candidate profile.
- Use the following structure (in this order):
  - Summary
  - Experience highlights
  - Key skills
  - Notable strengths
  - Suitable roles
- Use bullet points where appropriate.
- Keep the response concise but informative.

2) COMPARISON MODE
If the user asks about MULTIPLE resume IDs
(e.g. "compare Resume ID 57667857 and 11847784"):

- Output a clean Markdown table.
- Do NOT use HTML tags (e.g. <br>, <li>, <p>).
- Do NOT embed raw newlines inside table cells.
- Inside table cells:
  - Use short phrases.
  - Separate multiple items using commas or semicolons.
- Use consistent rows for all candidates.
- Recommended rows:
  - Role / Title
  - Years of experience
  - Key skills
  - Notable strengths
  - Best-fit roles or situations

- If a field is missing for a candidate, write:
  "not specified in the context".

=====================
STYLE REQUIREMENTS
=====================
- Prefer short, scannable phrases.
- One idea per bullet or phrase.
- Maintain consistent terminology across candidates.
- Optimize for readability in a Streamlit Markdown UI.
"""


def rag_agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not isinstance(last_message, HumanMessage):
        return {
            "messages": [],
            "token_usage": None,
            "route": state.get("route", "RAG_AGENT"),
            "selected_resume_ids": state.get("selected_resume_ids"),
}

    user_query = last_message.content
    selected_ids = state.get("selected_resume_ids") or []

    # Follow-up resolution
    lower_q = user_query.lower()

    if selected_ids:
        if "first" in lower_q and len(selected_ids) >= 1:
            user_query = f"Tell me more about Resume ID: {selected_ids[0]}"
        elif "second" in lower_q and len(selected_ids) >= 2:
            user_query = f"Tell me more about Resume ID: {selected_ids[1]}"
        elif "third" in lower_q and len(selected_ids) >= 3:
            user_query = f"Tell me more about Resume ID: {selected_ids[2]}"
        elif "that" in lower_q and len(selected_ids) == 1:
            user_query = f"Tell me more about Resume ID: {selected_ids[0]}"
  

    # 1. Call RAG tool
    rag_context = search_resumes.run(user_query)
    found_ids = extract_resume_ids(rag_context)


    # 2. Build prompt
    prompt_messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User question:\n{user_query}\n\n"
                f"Here is the resume context from the database:\n{rag_context}\n\n"
                "Use ONLY this information to answer the user's question."
            )
        ),
    ]

    response = rag_llm.invoke(prompt_messages)

    # response is an AIMessage and has usage_metadata in LangChain
    usage = getattr(response, "usage_metadata", None)

    # Update state
    new_state: AgentState = dict(state)
    new_state["messages"] = [response]
    new_state["token_usage"] = usage

    # ✅ MEMORY: store resume IDs for follow-up questions
    if found_ids:
        new_state["selected_resume_ids"] = found_ids
    else:
        new_state["selected_resume_ids"] = state.get("selected_resume_ids")

    return new_state





# =========================
# 6. CHAT AGENT NODE (GENERAL CHAT)
# =========================

CHAT_SYSTEM_PROMPT = """
You are a friendly and helpful general-purpose assistant.

Your responsibilities:
- Answer general questions that are not directly related to the resume dataset.
- If the user asks about specific resume details, skills, or candidate experience, guide them to ask clearly so the system can retrieve relevant resume information.
- Do NOT fabricate any specific candidate data without proper resume context.
"""

def chat_agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    input_messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)] + messages

    response = chat_llm.invoke(input_messages)
    usage = getattr(response, "usage_metadata", None)

    new_state: AgentState = dict(state)
    new_state["messages"] = [response]
    new_state["token_usage"] = usage

    # ✅ keep previous selected_resume_ids unchanged
    new_state["selected_resume_ids"] = state.get("selected_resume_ids")

    return new_state


# =========================
# 7. BUILD GRAPH
# =========================

def build_app():
    """
    Building and compiling the LangGraph app.
    """
    graph = StateGraph(AgentState)

    # Tambah node
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("RAG_AGENT", rag_agent_node)
    graph.add_node("CHAT_AGENT", chat_agent_node)

    # Alur:
    # start -> supervisor
    graph.set_entry_point("supervisor")

    # supervisor -> (RAG_AGENT | CHAT_AGENT) berdasarkan route_from_state
    graph.add_conditional_edges(
        "supervisor",
        route_from_state,
        {
            "RAG_AGENT": "RAG_AGENT",
            "CHAT_AGENT": "CHAT_AGENT",
        },
    )

    # Kedua agent selesai -> END
    graph.add_edge("RAG_AGENT", END)
    graph.add_edge("CHAT_AGENT", END)

    app = graph.compile()
    return app


# Global app (bisa langsung di-import di tempat lain, misalnya di Streamlit)
app = build_app()


# =========================
# 8. HELPER UNTUK UJI COBA
# =========================

def run_once(user_input: str, history: List[AnyMessage] | None = None) -> AIMessage:
    """
    A simple helper for testing graphs in plain Python.
    - user_input: user question text
    - history: list of previous messages (HumanMessage/AIMessage) (optional)
    """
    if history is None:
        history = []

    # Tambah pesan user ke history
    messages = history + [HumanMessage(content=user_input)]

    # Jalankan graph satu turn
    final_state = app.invoke({"messages": messages})

    # final_state["messages"] berisi semua pesan, termasuk jawaban terakhir
    all_messages = final_state["messages"]
    last_ai = None
    for m in reversed(all_messages):
        if isinstance(m, AIMessage):
            last_ai = m
            break

    return last_ai
