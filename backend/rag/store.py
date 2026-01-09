# backend/rag/store.py
import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from rag.ollama_client import embed_texts

# backend/rag/store.py  -> backend directory = .. (parent du dossier rag)
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_DIR = os.path.join(BACKEND_DIR, "storage", "chroma")

os.makedirs(CHROMA_DIR, exist_ok=True)

_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)

_collection = _client.get_or_create_collection(name="docs")


def _ids_for_file(file_name: str, n: int) -> List[str]:
    safe = file_name.replace("\\", "_").replace("/", "_")
    return [f"{safe}::chunk::{i}" for i in range(n)]


def delete_file_chunks(file_name: str) -> int:
    """
    Supprime tous les chunks indexés pour un fichier.
    Retourne le nombre d'items supprimés.
    """
    try:
        existing = _collection.get(where={"file_name": file_name})
        ids = existing.get("ids") or []
        if ids:
            _collection.delete(ids=ids)
            return len(ids)
    except Exception:
        pass
    return 0


def add_chunks(file_name: str, chunks: List[str]) -> int:
    chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
    if not chunks:
        return 0

    # overwrite par fichier
    delete_file_chunks(file_name)

    embs = embed_texts(chunks)
    ids = _ids_for_file(file_name, len(chunks))

    ext = os.path.splitext(file_name)[1].lower().lstrip(".")
    metadatas = []
    for i, c in enumerate(chunks):
        metadatas.append(
            {
                "file_name": file_name,
                "chunk_index": i,
                "doc_type": ext,
                "chunk_chars": len(c),
            }
        )

    _collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embs,
        metadatas=metadatas,
    )
    return len(chunks)


def query_top_k(query: str, k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Retourne une liste triée (best-first) de hits:
      {"doc": str, "meta": dict, "distance": float}
    where: filtre Chroma (ex: {"file_name": {"$in": ["a.pdf","b.pdf"]}})
    """
    q = (query or "").strip()
    if not q:
        return []

    k = max(1, int(k))

    q_emb = embed_texts([q])[0]
    res = _collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    if not dists:
        dists = [None] * len(docs)

    hits: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({"doc": doc, "meta": meta, "distance": dist})

    # best-first: distance plus petite = meilleure
    hits.sort(key=lambda h: (h["distance"] is None, h["distance"] if h["distance"] is not None else 10**9))
    return hits


def list_indexed_files() -> List[Dict[str, Any]]:
    """
    Liste les fichiers présents dans l'index + nombre de chunks.
    """
    try:
        res = _collection.get(include=["metadatas"])
        metas = res.get("metadatas") or []
        counts: Dict[str, int] = {}
        for m in metas:
            fn = (m or {}).get("file_name")
            if fn:
                counts[fn] = counts.get(fn, 0) + 1
        out = [{"file_name": k, "chunks": v} for k, v in counts.items()]
        out.sort(key=lambda x: x["file_name"].lower())
        return out
    except Exception:
        return []


def clear_index() -> int:
    """
    Vide complètement la collection. Retourne le nombre d'items supprimés.
    """
    try:
        res = _collection.get()
        ids = res.get("ids") or []
        if ids:
            _collection.delete(ids=ids)
            return len(ids)
    except Exception:
        pass
    return 0


def count_chunks() -> int:
    try:
        res = _collection.get()
        ids = res.get("ids") or []
        return len(ids)
    except Exception:
        return 0
