from __future__ import annotations

import os
import shutil
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# path where chroma will persist the vector store
_VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "/tmp/vectorstore")

_embeddings: OpenAIEmbeddings | HuggingFaceEmbeddings | None = None


def get_embeddings():
    """Return a singleton embeddings instance."""
    global _embeddings
    if _embeddings is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
        if not provider:
            provider = "openai" if os.getenv("OPENAI_API_KEY") else "hf"

        if provider == "openai":
            _embeddings = OpenAIEmbeddings()
        else:
            model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            _embeddings = HuggingFaceEmbeddings(model_name=model)
    return _embeddings


def _create_vectorstore_from_policy(policy_text: str) -> Chroma:
    """Initialize a fresh Chroma vector store from the raw policy text.

    The policy is split into small chunks (800 chars max) so that retrieval
    returns concise, targeted passages rather than bloated text.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(policy_text) if policy_text else []
    docs = [Document(page_content=c) for c in chunks]

    # Clean up stale vectorstore folder if it exists
    if os.path.exists(_VECTORSTORE_PATH):
        try:
            shutil.rmtree(_VECTORSTORE_PATH, ignore_errors=True)
        except Exception:
            pass

    vs = Chroma.from_documents(docs, get_embeddings(), persist_directory=_VECTORSTORE_PATH)
    return vs


def get_vectorstore(policy_text: str) -> Chroma:
    """Return a fresh vectorstore built from the provided policy text."""
    return _create_vectorstore_from_policy(policy_text)


def retrieve_policy_snippets(policy_text: str, query: str, k: int = 3) -> List[str]:
    """Run a similarity search against the policy store and return concise snippets."""
    if not policy_text.strip():
        return []
    vs = get_vectorstore(policy_text)
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]

