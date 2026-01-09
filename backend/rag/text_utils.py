import re
from typing import List

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Chunking amélioré:
    - normalise les sauts de lignes
    - split par paragraphes (prioritaire) puis découpe si trop long
    - overlap glissant entre chunks
    """
    if not isinstance(text, str):
        return []

    # Normalisation "douce" : garder les paragraphes, réduire l'excès de lignes
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return []

    chunk_size = max(200, int(chunk_size))
    overlap = max(0, int(overlap))
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    # 1) paragraphes
    parts = re.split(r"\n\n+", t)
    base_chunks: List[str] = []
    cur = ""

    def flush():
        nonlocal cur
        if cur and cur.strip():
            base_chunks.append(cur.strip())
        cur = ""

    for p in parts:
        p = p.strip()
        if not p:
            continue

        # si ajout dépasse la taille, flush
        if len(cur) + len(p) + 2 > chunk_size:
            flush()

            # si paragraphe trop long, découpe lignes puis fallback caractères
            if len(p) > chunk_size:
                lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
                buf = ""
                for ln in lines:
                    if len(buf) + len(ln) + 1 > chunk_size:
                        if buf:
                            base_chunks.append(buf.strip())
                        buf = ln
                    else:
                        buf = (buf + "\n" + ln).strip() if buf else ln

                if buf:
                    base_chunks.append(buf.strip())
            else:
                cur = p
        else:
            cur = (cur + "\n\n" + p).strip() if cur else p

    flush()

    # fallback si rien
    if not base_chunks:
        base_chunks = [t[:chunk_size]]

    # 2) overlap glissant
    if overlap <= 0:
        return [c for c in base_chunks if c.strip()]

    out: List[str] = []
    prev = ""
    for c in base_chunks:
        c = c.strip()
        if not c:
            continue
        if prev:
            prefix = prev[-overlap:]
            out.append((prefix + "\n" + c).strip())
        else:
            out.append(c)
        prev = c

    return out
