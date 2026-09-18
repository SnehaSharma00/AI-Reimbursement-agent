from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env
load_dotenv()

_llm_instance: Optional[ChatGroq] = None


def get_llm() -> ChatGroq:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            temperature=0,
        )
    return _llm_instance


def get_policy_retriever(policy_text: str):
    """Utility to obtain a LangChain retriever for a given reimbursement policy.

    This simply wraps the RAG helper so callers can get a retriever without
    importing the implementation directly.  The retriever will build/restore
    the underlying vector store on first use.
    """
    from src.utils.rag import get_vectorstore

    return get_vectorstore(policy_text).as_retriever()
