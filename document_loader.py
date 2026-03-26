import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_core.documents import Document

from config import config


def load_documents(docs_dir: str | None = None) -> list[Document]:
    """Load all PDF and CSV files from the documents directory."""
    docs_path = Path(docs_dir or config.docs_dir)
    documents: list[Document] = []

    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)
        return documents

    for file_path in docs_path.iterdir():
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
        elif file_path.suffix.lower() == ".csv":
            loader = CSVLoader(str(file_path))
            documents.extend(loader.load())

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
