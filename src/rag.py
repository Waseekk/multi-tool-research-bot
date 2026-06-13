"""
src/rag.py
==========
PDF processing and vector search for the Research Bot.

PDFProcessor         — extracts text, per-page content, and metadata from PDF bytes
ResearchVectorStore  — persistent ChromaDB collection with per-user PDF isolation

Storage layout:
  data/chroma_db/        ChromaDB on-disk store (auto-created)
  Collection name:       research_papers
  Chunk IDs:             {user_id}_{safe_filename}_{chunk_index}
  Metadata per chunk:    user_id, pdf_name, paper_title, page_num, chunk_id

First use downloads sentence-transformers/all-mpnet-base-v2 (~420 MB).
Subsequent uses load from the HuggingFace cache — fast after first run.
"""

import re
from pathlib import Path
from typing import Dict, List

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_PATH = Path("data/chroma_db")
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300


class PDFProcessor:
    """
    Extract text and metadata from PDF bytes using PyMuPDF (fitz).

    Each page is prefixed with a [Page N] marker so downstream chunking can
    track which physical page a chunk came from. The marker format is the same
    as the old pdf_chat project, so regex extraction works the same way.
    """

    def process_pdf(self, file_bytes: bytes, filename: str) -> Dict:
        """
        Parse a PDF and return structured data.

        Parameters
        ----------
        file_bytes : raw bytes from st.file_uploader or open(path, "rb")
        filename   : original file name, used as fallback title

        Returns
        -------
        {
            "text":          full text with [Page N] markers between pages,
            "pages_content": {page_number: text, ...}  (1-indexed),
            "metadata":      {"title": str, "authors": str, "pages": int},
            "filename":      str,
        }
        """
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF is not installed. Run: pip install PyMuPDF")

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(doc)

        # PDF metadata (may be empty for many research PDFs)
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip() or filename.replace(".pdf", "")
        authors = (meta.get("author") or "").strip() or "Unknown"

        pages_content: Dict[int, str] = {}
        text_parts: List[str] = []

        for page_idx in range(page_count):
            page = doc[page_idx]
            page_text = page.get_text().strip()
            page_num = page_idx + 1
            if page_text:
                pages_content[page_num] = page_text
                text_parts.append(f"[Page {page_num}]\n{page_text}")

        doc.close()

        return {
            "text": "\n\n".join(text_parts),
            "pages_content": pages_content,
            "metadata": {"title": title, "authors": authors, "pages": page_count},
            "filename": filename,
        }


class ResearchVectorStore:
    """
    ChromaDB-backed semantic search for uploaded research papers.

    Each user's PDFs are isolated by user_id metadata so searches never
    return another user's documents. Cosine similarity is used (better than
    L2 for high-dimensional text embeddings).

    Designed to be instantiated per-request — ChromaDB and the sentence-
    transformer model are both cached internally after first load, so
    repeated instantiation is cheap (no re-download, no re-init of the index).
    """

    def __init__(self):
        if not CHROMA_AVAILABLE:
            raise RuntimeError(
                "chromadb is not installed. Run: pip install chromadb sentence-transformers"
            )
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        # SentenceTransformerEmbeddingFunction downloads the model on first use
        # and caches it in the HuggingFace hub cache (~/.cache/huggingface/)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_pdf(self, pdf_data: Dict, user_id: str) -> int:
        """
        Chunk, embed, and persist a processed PDF in the vector store.

        Automatically deletes any existing chunks for the same (user, pdf_name)
        pair before adding, so re-uploading the same file is safe.

        Returns the number of chunks stored.
        """
        text = pdf_data["text"]
        filename = pdf_data["filename"]
        title = pdf_data["metadata"]["title"]

        if not text.strip():
            return 0

        # Remove stale chunks before re-indexing
        self.delete_pdf(filename, user_id)

        chunks = self._splitter.split_text(text)
        if not chunks:
            return 0

        # ChromaDB IDs must be unique strings with no spaces
        safe_name = re.sub(r"[^a-zA-Z0-9\-_]", "_", filename)
        ids = [f"{user_id}_{safe_name}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "user_id": user_id,
                "pdf_name": filename,
                "paper_title": title,
                "chunk_id": i,
                "page_num": self._extract_page_num(chunk),
            }
            for i, chunk in enumerate(chunks)
        ]

        self._collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def delete_pdf(self, pdf_name: str, user_id: str) -> None:
        """Remove all stored chunks for this (user, pdf) pair."""
        try:
            existing = self._collection.get(
                where={"$and": [{"user_id": user_id}, {"pdf_name": pdf_name}]}
            )
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(self, query: str, user_id: str, k: int = 6) -> List[Dict]:
        """
        Semantic search across this user's uploaded PDFs.

        Parameters
        ----------
        query   : natural-language search string
        user_id : only search documents belonging to this user
        k       : max results to return

        Returns
        -------
        List of dicts with keys: text, pdf_name, paper_title, page_num, score
        """
        # Count user's documents to avoid querying an empty set (ChromaDB errors on n_results=0)
        try:
            user_data = self._collection.get(where={"user_id": user_id})
        except Exception:
            return []

        n_user_docs = len(user_data["ids"])
        if n_user_docs == 0:
            return []

        n_results = min(k, n_user_docs)
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"user_id": user_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc,
                "pdf_name": meta.get("pdf_name", ""),
                "paper_title": meta.get("paper_title", ""),
                "page_num": meta.get("page_num", 0),
                # Convert cosine distance [0, 2] → similarity score [0, 1]
                "score": round(max(0.0, 1.0 - dist / 2), 3),
            })
        return output

    def list_pdfs(self, user_id: str) -> List[str]:
        """Return sorted list of unique PDF names stored for a user."""
        try:
            result = self._collection.get(where={"user_id": user_id})
            return sorted({m["pdf_name"] for m in result["metadatas"]})
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_page_num(chunk: str) -> int:
        """Return the last [Page N] number found in a chunk (0 if none)."""
        matches = re.findall(r"\[Page (\d+)\]", chunk)
        return int(matches[-1]) if matches else 0
