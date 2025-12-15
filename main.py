import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from agents_graph import app as graph_app  # LangGraph compiled app

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Resume RAG Agent", page_icon="📄")

st.title("📄 Resume RAG Chatbot")
st.caption("Multi-agent RAG over resume dataset (LangGraph + Qdrant + OpenAI)")

MAX_HISTORY = 10  # last 10 messages (user+assistant)

# =========================
# 1. SESSION STATE SETUP
# =========================

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "lc_messages" not in st.session_state:
    st.session_state["lc_messages"] = []

if "token_usage_total" not in st.session_state:
    st.session_state["token_usage_total"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

if "graph_state" not in st.session_state:
    st.session_state["graph_state"] = {
        "messages": [],
        "route": "CHAT_AGENT",
        "token_usage": None,
        "selected_resume_ids": None,
    }

# =========================
# 2. SIDEBAR: TOKEN USAGE
# =========================

with st.sidebar:
    st.header("📊 LLM Usage")

    usage_total = st.session_state["token_usage_total"]
    st.write(f"**Input tokens:** {usage_total['input_tokens']}")
    st.write(f"**Output tokens:** {usage_total['output_tokens']}")
    st.write(f"**Total tokens:** {usage_total['total_tokens']}")

    st.markdown("---")

    if st.button("🔁 Clear conversation"):
        st.session_state["chat_history"] = []
        st.session_state["lc_messages"] = []
        st.session_state["token_usage_total"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        st.session_state["graph_state"] = {
            "messages": [],
            "route": "CHAT_AGENT",
            "token_usage": None,
            "selected_resume_ids": None,
        }
        st.rerun()

# =========================
# 3. DISPLAY CHAT HISTORY
# =========================

for msg in st.session_state["chat_history"]:
    role = msg["role"]
    content = msg["content"]
    source = msg.get("source")

    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)
        if role == "assistant" and source:
            st.caption(f"Answer source: **{source}**")

# =========================
# 4. CHAT INPUT
# =========================

user_input = st.chat_input("Ask something about candidates, resumes, or general topics...")

if user_input:
    # Show user message immediately
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # Add to LangChain history
    lc_messages = st.session_state["lc_messages"]
    lc_messages.append(HumanMessage(content=user_input))

    # Short memory window for the graph
    history_window = lc_messages[-MAX_HISTORY:]

    # Build state for graph invoke (preserve selected_resume_ids etc.)
    current_state = st.session_state["graph_state"]
    current_state.setdefault("route", "CHAT_AGENT")
    current_state.setdefault("token_usage", None)
    current_state.setdefault("selected_resume_ids", None)
    current_state["messages"] = history_window

    final_state = graph_app.invoke(current_state)

    # Persist graph state (keeps selected_resume_ids memory)
    st.session_state["graph_state"] = final_state

    # Find last AI message
    all_messages = final_state.get("messages", [])
    last_ai = next((m for m in reversed(all_messages) if isinstance(m, AIMessage)), None)

    if last_ai is None:
        assistant_reply = "Sorry, I could not generate a response."
        source_label = "Unknown"
    else:
        assistant_reply = last_ai.content

        route = final_state.get("route", "CHAT_AGENT")
        source_label = "RAG Agent (Resume Database)" if route == "RAG_AGENT" else "Chat Agent (General Knowledge)"

        # Append AI to full UI history
        lc_messages.append(last_ai)

    # Token usage update (only if non-empty)
    usage = final_state.get("token_usage")
    if isinstance(usage, dict) and any(usage.values()):
        st.session_state["token_usage_total"]["input_tokens"] += usage.get("input_tokens", 0)
        st.session_state["token_usage_total"]["output_tokens"] += usage.get("output_tokens", 0)
        st.session_state["token_usage_total"]["total_tokens"] += usage.get("total_tokens", 0)

    # Display assistant reply
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
        st.caption(f"Answer source: **{source_label}**")

    st.session_state["chat_history"].append(
        {"role": "assistant", "content": assistant_reply, "source": source_label}
    )
