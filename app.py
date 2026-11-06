"""
Streamlit UI for the arXiv RAG assistant.

Run:
    streamlit run app.py

Keep this file in the same folder as rag_pipeline.py (and chroma_arxiv/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="arXiv Research Expert", page_icon="📚", layout="centered")

# Draw something immediately so the page is never blank while heavy imports run.
st.title("📚 arXiv Research Expert")
st.caption(
    "Ask a narrowed down question about machine learning, deep learning, NLP, AI, "
    "LLMs, or other recent research topics. Answers are grounded in the "
    "latest research papers."
)

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

with st.spinner("Importing libraries (first load can take a minute)..."):
    try:
        from langchain_ollama import ChatOllama
        import rag_pipeline as rp
    except Exception as exc:
        st.error(f"Could not import rag_pipeline.py: {exc!r}")
        st.exception(exc)
        st.stop()



#load heavy objects once per server process, not on every rerun.


@st.cache_resource(show_spinner="Loading vector store, models and reranker...")
def load_pipeline():
    vectorstore = rp.create_vectorstore()

    # Models are loaded lazily by Ollama on the first question.
    # (Add keep_alive="30m" to ChatOllama later if you want them to stay loaded.)
    llm = ChatOllama(model=rp.OLLAMA_MODEL, temperature=0)
    filter_llm = ChatOllama(
        model=rp.FILTER_MODEL, temperature=0
    ).with_structured_output(rp.SearchFilters, method="json_schema")

    rp.get_reranker()  # local CPU model, safe to load at startup
    return vectorstore, llm, filter_llm


try:
    vectorstore, llm, filter_llm = load_pipeline()
except Exception as exc:  # missing chroma dir, ollama down, etc.
    st.error(f"Could not start the pipeline: {exc!r}")
    st.exception(exc)
    st.stop()



# Sidebar: optional manual filters (the LLM also extracts filters from the
# question itself, and the two are combined).

with st.sidebar:
    st.header("Optional filters")
    st.caption("Filters are also detected automatically from your question.")

    categories = st.multiselect(
        "Categories",
        ["cs_AI", "cs_CL", "cs_CV", "cs_LG", "cs_NE", "stat_ML"],
    )

    use_years = st.checkbox("Filter by year")
    year_from = year_to = None
    if use_years:
        year_from, year_to = st.slider("Published between", 2007, 2026, (2023, 2026))

    only_llm = st.checkbox("Only papers mentioning LLMs")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

manual_filter = rp.build_metadata_filter(
    categories=categories or None,
    year_from=year_from,
    year_to=year_to,
    mentions_llm=True if only_llm else None,
)



# Chat UI

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)} papers)"):
        for i, s in enumerate(sources, start=1):
            title = f"[{s['title']}]({s['url']})" if s["url"] else s["title"]
            st.markdown(f"**{i}. {title}**")
            st.caption(f"{s['authors']} · {s['published']}")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources", []))


if question := st.chat_input("Ask about recent research..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Phase 1: filters + retrieval + rerank (not streamable, so show status).
        with st.status("Searching papers...", expanded=False) as status:
            try:
                context, results, _ = rp.prepare_context(
                    vectorstore, filter_llm, question, manual_filter
                )
            except Exception as exc:
                status.update(label="Search failed", state="error")
                st.error(f"Error: {exc}")
                st.stop()
            status.update(
                label=f"Found {len(results)} relevant paper(s)", state="complete"
            )

        # Phase 2: stream the answer word by word.
        if context is None:
            answer = rp.NO_KNOWLEDGE
            st.markdown(answer)
        else:
            try:
                answer = st.write_stream(rp.stream_answer(llm, context, question))
            except Exception as exc:
                st.error(f"Answer generation failed: {exc}")
                st.stop()

        sources = [
            {
                "title": d.metadata.get("title", "Untitled paper"),
                "authors": d.metadata.get("authors", "Authors not listed"),
                "published": d.metadata.get("published", d.metadata.get("year", "")),
                "url": d.metadata.get("url", ""),
                "rerank": d.metadata.get("rerank_score", float("nan")),
                "distance": dist,
            }
            for d, dist in results
        ]
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )