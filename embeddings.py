from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Return a sentence-transformer embedding function.

    Uses all-MiniLM-L6-v2 by default — small, fast, free, runs on CPU.
    """
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
