from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from config import config
from embeddings import get_embedding_function
from document_loader import load_documents, chunk_documents


PROMPT_TEMPLATE = """You are a financial document analyst. Answer the question based ONLY on the following context from financial documents. If the answer is not in the context, say "I don't have enough information in the loaded documents to answer that."

Context:
{context}

Question: {question}

Provide a clear, concise answer. Always cite the source document name and page number when available.

Answer:"""


class RAGEngine:
    def __init__(self):
        self.embedding_fn = get_embedding_function()
        self.vector_store: Chroma | None = None
        self.qa_chain = None

    def ingest(self, docs_dir: str | None = None) -> int:
        """Load documents, chunk them, and store embeddings. Returns chunk count."""
        documents = load_documents(docs_dir)
        if not documents:
            return 0

        chunks = chunk_documents(documents)
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_fn,
            persist_directory=config.chroma_persist_dir,
        )
        self._build_qa_chain()
        return len(chunks)

    def load_existing(self) -> bool:
        """Load previously ingested vector store from disk."""
        try:
            self.vector_store = Chroma(
                persist_directory=config.chroma_persist_dir,
                embedding_function=self.embedding_fn,
            )
            if self.vector_store._collection.count() == 0:
                return False
            self._build_qa_chain()
            return True
        except Exception:
            return False

    def query(self, question: str) -> dict:
        """Ask a question and get an answer with source documents."""
        if not self.qa_chain:
            return {"answer": "No documents loaded. Please ingest documents first.", "sources": []}

        result = self.qa_chain.invoke({"query": question})
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:200],
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
            })

        return {"answer": result["result"], "sources": sources}

    def _build_qa_chain(self):
        """Wire up the retrieval QA chain with the configured LLM."""
        llm = self._get_llm()
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def _get_llm(self):
        if config.llm_provider == "openai":
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=config.openai_api_key)
        return Ollama(model=config.ollama_model, temperature=0)
