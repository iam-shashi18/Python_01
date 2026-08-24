"""
Day 7 (Bonus) - LangChain edition of the Enterprise RAG Chatbot
----------------------------------------------------------------
Same endpoints as Day 6, rewritten with LangChain + LCEL.

Requires TOGETHER_API_KEY in .env.

Start:
    uvicorn main:app --reload
Open http://localhost:8000/docs

Endpoints:
    POST /ingest   { "source": "path or URL" }
    POST /ask      { "question": "...", "top_k": 5 }         (streams)
    GET  /usage

This build is open - there is no authentication. Every request shares one
identity, so don't expose it outside a trusted network.
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- LangChain imports ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together import ChatTogether

load_dotenv()


# ------------------------------------------------------------
# Setup — one-time at startup
# ------------------------------------------------------------
# No auth: every request is attributed to this single identity, which is what
# document metadata and the usage table are keyed on.
DEFAULT_USER = "local"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker_model, top_n=5)

vstore = Chroma(
    collection_name="langchain_enterprise_kb",
    embedding_function=embeddings,
    persist_directory="./langchain_rag_db",
)

llm = ChatTogether(model="openai/gpt-oss-20b", temperature=0.2)
enc = tiktoken.encoding_for_model("gpt-4o-mini")

conn = sqlite3.connect("./langchain_usage.db", check_same_thread=False)
conn.execute(
    "CREATE TABLE IF NOT EXISTS usage("
    "user TEXT, ts TEXT, input_tokens INT, output_tokens INT)"
)
conn.commit()

app = FastAPI(title="Enterprise RAG Chatbot — LangChain edition")


# ------------------------------------------------------------
# Loaders — dispatch to the right LangChain loader
# ------------------------------------------------------------
def load(source: str):
    if source.startswith("http"):
        return WebBaseLoader(source).load()
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(source).load()
    if ext == ".docx":
        return Docx2txtLoader(source).load()
    if ext in {".md", ".txt"}:
        return TextLoader(source, encoding="utf-8").load()
    raise ValueError(f"Unsupported format: {ext}")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ------------------------------------------------------------
# Guardrails
# ------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (all |any )?(previous|prior|above)",
    r"you are now [A-Z]",
    r"reveal (the |your )?(system|initial) prompt",
]


def looks_like_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)


# ------------------------------------------------------------
# Prompt + LCEL chain (built lazily per-request so we can attach
# per-user retriever + track tokens)
# ------------------------------------------------------------
SYSTEM = (
    "You are a helpful assistant. Answer using ONLY the numbered context. "
    "If the answer is not there, say 'I don\'t know.' "
    "Cite sources with bracket numbers.\n\n"
    "Format your answer like this:\n"
    "The Pro plan costs $29/month [2]."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user",   "Context:\n{context}\n\n<user_question>\n{question}\n</user_question>"),
])


def format_docs(docs) -> str:
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))


def build_rag_chain(user: str):
    """Retriever filtered to the current user's docs + rerank + LLM."""
    base_retriever = vstore.as_retriever(
        search_kwargs={"k": 20, "filter": {"user": user}},
    )
    reranked = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )
    return (
        {"context":  reranked | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
class IngestBody(BaseModel):
    source: str


@app.post("/ingest")
def ingest(body: IngestBody):
    user = DEFAULT_USER
    docs = load(body.source)
    if not docs:
        return {"chunks_added": 0}

    for d in docs:
        d.metadata["user"] = user
        d.metadata["source"] = d.metadata.get("source", body.source)

    chunks = splitter.split_documents(docs)
    if not chunks:
        return {"chunks_added": 0}

    vstore.add_documents(chunks)
    return {"chunks_added": len(chunks), "source": body.source}


class AskBody(BaseModel):
    question: str
    top_k: int = 5


def log_usage(user: str, in_tok: int, out_tok: int) -> None:
    conn.execute(
        "INSERT INTO usage VALUES (?, ?, ?, ?)",
        (user, datetime.utcnow().isoformat(), in_tok, out_tok),
    )
    conn.commit()


def stream_answer(user: str, question: str):
    """Stream tokens from the LCEL chain and log usage at the end."""
    chain = build_rag_chain(user)
    in_tokens = len(enc.encode(SYSTEM)) + len(enc.encode(question))
    out_text = ""
    for piece in chain.stream(question):
        out_text += piece
        if piece:
            yield piece
    log_usage(user, in_tokens, len(enc.encode(out_text)))


@app.post("/ask")
def ask(body: AskBody):
    user = DEFAULT_USER
    if looks_like_injection(body.question):
        return {"answer": "I can\'t help with that."}

    # Distance-threshold-style refusal - LangChain doesn't expose scores
    # directly on Retriever, so peek at the base similarity search first.
    hits = vstore.similarity_search_with_score(
        body.question, k=1, filter={"user": user},
    )
    # Higher score = more similar for HuggingFaceEmbeddings via Chroma.
    # Chroma actually returns distances (lower = better) with default settings,
    # so refuse when top distance > threshold.
    if not hits or hits[0][1] > 1.0:
        return {"answer": "I don\'t know - I couldn\'t find a confident match."}

    return StreamingResponse(
        stream_answer(user, body.question),
        media_type="text/plain",
    )


@app.get("/usage")
def usage():
    user = DEFAULT_USER
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) "
        "FROM usage WHERE user = ?",
        (user,),
    ).fetchone()
    n, in_tok, out_tok = row
    # openai/gpt-oss-20b on Together: $0.05/M in, $0.20/M out
    cost = (in_tok / 1_000_000) * 0.05 + (out_tok / 1_000_000) * 0.20
    return {
        "user": user,
        "questions": n,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "est_cost_usd": round(cost, 6),
    }
