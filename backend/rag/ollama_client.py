# backend/rag/ollama_client.py
import os
import requests
from typing import List

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def embed_texts(texts: List[str], base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_EMBED_MODEL) -> List[List[float]]:
    """
    Retourne une liste d'embeddings (List[float]) pour chaque texte.
    Compatible avec /api/embed (nouveau) et /api/embeddings (ancien).
    """
    texts = [t if isinstance(t, str) else str(t) for t in texts]
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return []

    # 1) Essaye /api/embed (moderne): {model, input:[...]} -> {embeddings:[...]}
    try:
        data = _post(f"{base_url}/api/embed", {"model": model, "input": texts}, timeout=120)
        embs = data.get("embeddings")
        if isinstance(embs, list) and embs and isinstance(embs[0], list):
            return embs
    except Exception:
        pass

    # 2) Fallback /api/embeddings (ancien): {model, prompt:"..."} -> {embedding:[...]} (un par un)
    out: List[List[float]] = []
    for t in texts:
        data = _post(f"{base_url}/api/embeddings", {"model": model, "prompt": t}, timeout=120)
        emb = data.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError("Ollama embeddings: format inattendu.")
        out.append(emb)
    return out


def chat_answer(system: str, user_question: str, context: str, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_CHAT_MODEL) -> str:
    # 1) /api/chat
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"SOURCES:\n{context}\n\nQUESTION: {user_question}\n\nRéponse:"},
        ]
        data = _post(f"{base_url}/api/chat", {"model": model, "messages": messages, "stream": False}, timeout=120)
        msg = data.get("message", {})
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    except Exception:
        pass

    # 2) fallback /api/generate
    prompt = f"{system}\n\nSOURCES:\n{context}\n\nQUESTION: {user_question}\n\nRéponse:"
    data = _post(f"{base_url}/api/generate", {"model": model, "prompt": prompt, "stream": False}, timeout=120)
    resp = data.get("response", "")
    return resp.strip() if isinstance(resp, str) else str(resp)
