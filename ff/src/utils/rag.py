from __future__ import annotations

import os
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# path where chroma will persist the vector store
# Use /tmp on Railway (ephemeral, always writable); override via VECTORSTORE_PATH env var
_VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "/tmp/vectorstore")

_embeddings: OpenAIEmbeddings | None = None
_vectorstore: Chroma | None = None


def get_embeddings():
    """Return a singleton embeddings instance.

    The project defaults to using OpenAI if ``OPENAI_API_KEY`` is present,
    but falls back to a HuggingFace model otherwise so that a key is **not**
    required.  You can also force a provider by setting ``EMBEDDING_PROVIDER``
    to ``openai`` or ``hf``.
    """
    global _embeddings
    if _embeddings is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
        if not provider:
            provider = "openai" if os.getenv("OPENAI_API_KEY") else "hf"

        if provider == "openai":
            _embeddings = OpenAIEmbeddings()
        else:
            # default HF model; modify via EMBEDDING_MODEL env var
            model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            _embeddings = HuggingFaceEmbeddings(model_name=model)
    return _embeddings


def _create_vectorstore_from_policy(policy_text: str) -> Chroma:
    """Initialize a Chroma vector store from the raw policy text.

    The policy is split into paragraphs and each paragraph is added as a
    separate document so that retrieval can return the most relevant passage.
    """
    # simple paragraph splitter
    chunks = [p.strip() for p in policy_text.split("\n\n") if p.strip()]
    docs = [Document(page_content=chunk) for chunk in chunks]

    vs = Chroma.from_documents(docs, get_embeddings(), persist_directory=_VECTORSTORE_PATH)
    vs.persist()
    return vs


def get_vectorstore(policy_text: str) -> Chroma:
    """Return a vectorstore, creating it if necessary.

    If a persisted store already exists on disk it will be reloaded; otherwise
    the provided policy text is used to build the store from scratch.
    """
    global _vectorstore
    if _vectorstore is None:
        if os.path.exists(_VECTORSTORE_PATH) and any(os.scandir(_VECTORSTORE_PATH)):
            _vectorstore = Chroma(persist_directory=_VECTORSTORE_PATH, embedding_function=get_embeddings())
        else:
            _vectorstore = _create_vectorstore_from_policy(policy_text)
    return _vectorstore


def retrieve_policy_snippets(policy_text: str, query: str, k: int = 3) -> List[str]:
    """Run a similarity search against the policy store and return textual snippets.

    Parameters
    ----------
    policy_text
        The full policy document used to build the store if it does not exist yet.
    query
        A text query (for example, the expense descriptions or categories) that
        will be used to search the store.
    k
        Number of top matching passages to return.
    """
    vs = get_vectorstore(policy_text)
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]
