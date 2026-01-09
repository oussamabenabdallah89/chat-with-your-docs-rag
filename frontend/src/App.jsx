// src/App.jsx
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const STORAGE_KEY = "rag_chat_history_v1";
const SELECTED_KEY = "rag_selected_files_v1";

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function fetchWithTimeout(url, options = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(id));
}

export default function App() {
  // Upload
  const [file, setFile] = useState(null);
  const [uploadMsg, setUploadMsg] = useState("");
  const [uploading, setUploading] = useState(false);

  // Docs
  const [docs, setDocs] = useState([]); // [{file_name, chunks}]
  const [docsLoading, setDocsLoading] = useState(false);

  // ✅ persist selected files
  const [selectedFiles, setSelectedFiles] = useState(() => {
    try {
      const raw = localStorage.getItem(SELECTED_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  });

  // ✅ search in docs list
  const [docSearch, setDocSearch] = useState("");

  // Chat
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // ✅ backend health
  const [backendOk, setBackendOk] = useState(true);
  const [chunksIndexed, setChunksIndexed] = useState(null);

  // messages: [{id, role:'user'|'assistant'|'system', content, sources?, ts}]
  const [messages, setMessages] = useState([]);
  const messagesRef = useRef([]);
  const endRef = useRef(null);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // load chat history
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) setMessages(JSON.parse(raw));
    } catch {
      // ignore
    }
  }, []);

  // persist chat history
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
      // ignore
    }
  }, [messages]);

  // ✅ persist selected files
  useEffect(() => {
    try {
      localStorage.setItem(SELECTED_KEY, JSON.stringify(selectedFiles));
    } catch {
      // ignore
    }
  }, [selectedFiles]);

  // autoscroll
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // load docs + health at start
  useEffect(() => {
    refreshHealth();
    refreshDocs();
    const t = setInterval(refreshHealth, 8000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const canSend = useMemo(() => input.trim().length > 0 && !loading && backendOk, [input, loading, backendOk]);

  const filteredDocs = useMemo(() => {
    const q = (docSearch || "").trim().toLowerCase();
    if (!q) return docs;
    return docs.filter((d) => (d.file_name || "").toLowerCase().includes(q));
  }, [docs, docSearch]);

  async function refreshHealth() {
    try {
      const res = await fetchWithTimeout(`${API}/health`, {}, 5000);
      if (!res.ok) throw new Error("health not ok");
      const data = await res.json();
      setBackendOk(true);
      setChunksIndexed(typeof data.chunks_indexed === "number" ? data.chunks_indexed : null);
    } catch {
      setBackendOk(false);
      setChunksIndexed(null);
    }
  }

  async function refreshDocs() {
    setDocsLoading(true);
    try {
      const res = await fetchWithTimeout(`${API}/documents`, {}, 10000);
      const data = await res.json();
      const files = data.files || [];
      setDocs(files);

      // Nettoie la sélection si des docs ont disparu
      setSelectedFiles((prev) => prev.filter((f) => files.some((d) => d.file_name === f)));
    } catch {
      // ignore
    } finally {
      setDocsLoading(false);
    }
  }

  function toggleSelected(fileName) {
    setSelectedFiles((prev) => {
      if (prev.includes(fileName)) return prev.filter((x) => x !== fileName);
      return [...prev, fileName];
    });
  }

  function clearSelection() {
    setSelectedFiles([]);
  }

  function selectAll() {
    setSelectedFiles(docs.map((d) => d.file_name));
  }

  // ✅ Select only this
  function selectOnly(fileName) {
    setSelectedFiles([fileName]);
  }

  async function uploadFile() {
    if (!file || uploading) return;

    setUploading(true);
    setUploadMsg("Uploading...");
    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetchWithTimeout(
        `${API}/documents/upload`,
        { method: "POST", body: form },
        60000
      );

      if (!res.ok) {
        const txt = await res.text();
        setUploadMsg(`Upload failed: ${txt}`);
        return;
      }

      const data = await res.json();
      setUploadMsg(`Uploaded: ${data.file_name} (chunks: ${data.chunks_indexed})`);

      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "system",
          content: `Document indexé: ${data.file_name} (chunks: ${data.chunks_indexed})`,
          ts: Date.now(),
        },
      ]);

      await refreshDocs();
      await refreshHealth();
    } catch (e) {
      const msg = e?.name === "AbortError" ? "Timeout upload (backend ne répond pas)." : String(e);
      setUploadMsg(`Upload failed: ${msg}`);
      await refreshHealth();
    } finally {
      setUploading(false);
    }
  }

  async function deleteDoc(fileName) {
    if (!fileName) return;
    const ok = window.confirm(`Supprimer "${fileName}" de l'index ?`);
    if (!ok) return;

    try {
      const res = await fetchWithTimeout(
        `${API}/documents/${encodeURIComponent(fileName)}`,
        { method: "DELETE" },
        15000
      );

      if (!res.ok) {
        const txt = await res.text();
        setMessages((prev) => [
          ...prev,
          { id: uid(), role: "system", content: `Erreur suppression: ${txt}`, ts: Date.now() },
        ]);
        return;
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "system",
          content: `Document supprimé: ${data.file_name} (chunks supprimés: ${data.chunks_deleted}).`,
          ts: Date.now(),
        },
      ]);

      setSelectedFiles((prev) => prev.filter((x) => x !== fileName));
      await refreshDocs();
      await refreshHealth();
    } catch (e) {
      const msg = e?.name === "AbortError" ? "Timeout suppression (backend ne répond pas)." : String(e);
      setMessages((prev) => [
        ...prev,
        { id: uid(), role: "system", content: `Erreur suppression: ${msg}`, ts: Date.now() },
      ]);
      await refreshHealth();
    }
  }

  async function clearIndex() {
    const ok = window.confirm("Vider tout l'index (tous les documents) ?");
    if (!ok) return;

    try {
      const res = await fetchWithTimeout(`${API}/index/clear`, { method: "POST" }, 20000);

      if (!res.ok) {
        const txt = await res.text();
        setMessages((prev) => [
          ...prev,
          { id: uid(), role: "system", content: `Erreur clear index: ${txt}`, ts: Date.now() },
        ]);
        return;
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "system",
          content: `Index vidé (chunks supprimés: ${data.chunks_deleted}).`,
          ts: Date.now(),
        },
      ]);

      setSelectedFiles([]);
      await refreshDocs();
      await refreshHealth();
    } catch (e) {
      const msg = e?.name === "AbortError" ? "Timeout clear index (backend ne répond pas)." : String(e);
      setMessages((prev) => [
        ...prev,
        { id: uid(), role: "system", content: `Erreur clear index: ${msg}`, ts: Date.now() },
      ]);
      await refreshHealth();
    }
  }

  // ✅ envoie l'historique + selected_files
  async function send() {
    const q = input.trim();
    if (!q || loading || !backendOk) return;

    const userMsg = { id: uid(), role: "user", content: q, ts: Date.now() };

    const prevMessages = messagesRef.current || [];
    const nextMessages = [...prevMessages, userMsg];

    setMessages(nextMessages);
    setInput("");
    setLoading(true);

    const history = nextMessages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .slice(-12)
      .map((m) => ({ role: m.role, content: m.content }));

    const selected_files = selectedFiles.length > 0 ? selectedFiles : null;

    try {
      const res = await fetchWithTimeout(
        `${API}/chat`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: q,
            top_k: 5,
            history,
            selected_files,
          }),
        },
        30000
      );

      if (!res.ok) {
        const txt = await res.text();
        setMessages((prev) => [
          ...prev,
          { id: uid(), role: "assistant", content: `Error: ${txt}`, sources: [], ts: Date.now() },
        ]);
        return;
      }

      const data = await res.json();
      const a = data.answer || "";
      const s = data.sources || [];

      setMessages((prev) => [
        ...prev,
        { id: uid(), role: "assistant", content: a, sources: s, ts: Date.now() },
      ]);

      await refreshHealth();
    } catch (e) {
      const msg = e?.name === "AbortError" ? "Timeout: backend ne répond pas." : String(e);
      setMessages((prev) => [
        ...prev,
        { id: uid(), role: "assistant", content: `Error: ${msg}`, sources: [], ts: Date.now() },
      ]);
      await refreshHealth();
    } finally {
      setLoading(false);
    }
  }

  function resetChat() {
    setMessages([]);
    setInput("");
    setUploadMsg("");
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  const scopeLabel =
    selectedFiles.length === 0
      ? "Recherche: tous les documents"
      : `Recherche: ${selectedFiles.length} doc(s) sélectionné(s)`;

  const healthLabel = backendOk
    ? `Backend: OK${typeof chunksIndexed === "number" ? ` • chunks: ${chunksIndexed}` : ""}`
    : "Backend: DOWN";

  return (
    <div className="app">
      {/* ✅ Robot à droite */}
      <img className="robotSide" src="/robot.png" alt="" aria-hidden="true" />

      <header className="topbar">
        <div className="brand">
          <h1>Chat with your Docs (RAG)</h1>
          <div className="subtitle">Local • FastAPI + Chroma + Ollama + React</div>
        </div>

        <div className="topbarActions">
          <button className="btn ghost" onClick={refreshDocs} disabled={docsLoading}>
            {docsLoading ? "Refreshing…" : "Refresh docs"}
          </button>
          <button className="btn ghost" onClick={resetChat} title="Vider l'historique">
            New chat / Reset
          </button>
        </div>
      </header>

      <main className="layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="panel">
            <div className="panelTitle">Documents</div>

            <div className="block">
              <div className="label">Upload document</div>
              <input
                className="file"
                type="file"
                accept=".txt,.pdf,.docx,.md"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />

              <button className="btn" onClick={uploadFile} disabled={!file || uploading}>
                {uploading ? "Uploading..." : "Upload"}
              </button>

              <div className="hint">{uploadMsg || "Upload un document, puis pose une question dans le chat."}</div>
            </div>

            <div className="divider" />

            {/* À propos */}
            <div className="mini">
              <div className="miniTitle">À propos</div>
              <div className="miniText">
                Ce projet te permet de discuter avec tes documents (PDF/DOCX/TXT/MD) grâce à un pipeline RAG local :
                le backend extrait et découpe le texte, l’indexe dans Chroma, puis Ollama génère une réponse basée
                uniquement sur les passages retrouvés.
              </div>
            </div>

            <div className="divider" />

            {/* Index actions */}
            <div className="mini">
              <div className="miniTitle">Index (Chroma)</div>
              <button className="btn danger" onClick={clearIndex}>
                Clear index
              </button>
              <div className="hint">{scopeLabel}</div>
            </div>

            <div className="divider" />

            {/* ✅ Health */}
            <div className="mini">
              <div className="miniTitle">Status</div>
              <div className={`hint ${backendOk ? "" : "hintError"}`}>{healthLabel}</div>
            </div>

            <div className="divider" />

            {/* Docs list + selection */}
            <div className="docsHeader">
              <div className="miniTitle">Indexed documents</div>
              <div className="docsCount">{docs.length} file(s)</div>
            </div>

            {/* ✅ Search box */}
            <input
              className="docSearch"
              placeholder="Rechercher un document…"
              value={docSearch}
              onChange={(e) => setDocSearch(e.target.value)}
            />

            <div className="docsTools">
              <button className="btn ghost" onClick={selectAll} disabled={docs.length === 0}>
                Select all
              </button>
              <button className="btn ghost" onClick={clearSelection} disabled={selectedFiles.length === 0}>
                Select none
              </button>
            </div>

            {docs.length === 0 ? (
              <div className="hint">Aucun document indexé pour le moment.</div>
            ) : filteredDocs.length === 0 ? (
              <div className="hint">Aucun document ne correspond à la recherche.</div>
            ) : (
              <div className="docsList">
                {filteredDocs.map((d) => {
                  const checked = selectedFiles.includes(d.file_name);
                  return (
                    <div className="docRow" key={d.file_name}>
                      <label className="docLeft">
                        <input
                          type="checkbox"
                          className="docCheck"
                          checked={checked}
                          onChange={() => toggleSelected(d.file_name)}
                        />
                        <div className="docInfo">
                          <div className="docName" title={d.file_name}>
                            {d.file_name}
                          </div>
                          <div className="docMeta">{d.chunks} chunks</div>
                        </div>
                      </label>

                      <div className="docActions">
                        {/* ✅ Select only this */}
                        <button className="btn ghost miniBtn" onClick={() => selectOnly(d.file_name)} title="Select only this">
                          Only
                        </button>
                        <button className="btn danger" onClick={() => deleteDoc(d.file_name)}>
                          Delete
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            <div className="api">API: {API}</div>
          </div>
        </aside>

        {/* Chat */}
        <section className="chat">
          <div className="chatWindow">
            {messages.length === 0 ? (
              <div className="empty">
                <div className="emptyCard">
                  <div className="emptyTitle">Pose une question après upload</div>
                  <div className="emptyText">Astuce: coche un ou plusieurs documents pour limiter la recherche.</div>
                </div>
              </div>
            ) : (
              <div className="messages">
                {messages.map((m) => (
                  <Message key={m.id} msg={m} />
                ))}
                {loading && (
                  <div className="msgRow assistant">
                    <div className="bubble assistant">
                      <span className="typing">Thinking…</span>
                    </div>
                  </div>
                )}
                <div ref={endRef} />
              </div>
            )}
          </div>

          <div className="composer">
            <textarea
              className="textarea"
              placeholder={
                backendOk
                  ? "Écris ta question… (Entrée = envoyer, Shift+Entrée = nouvelle ligne)"
                  : "Backend DOWN: impossible d’envoyer."
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={2}
              disabled={!backendOk}
            />
            <button className="btn send" onClick={send} disabled={!canSend}>
              Send
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}

function Message({ msg }) {
  if (msg.role === "system") {
    return (
      <div className="systemLine">
        <span className="systemBadge">INFO</span>
        <span className="systemText">{msg.content}</span>
      </div>
    );
  }

  const isUser = msg.role === "user";
  return (
    <div className={`msgRow ${isUser ? "user" : "assistant"}`}>
      <div className={`bubble ${isUser ? "user" : "assistant"}`}>
        <div className="content">{msg.content}</div>

        {!isUser && Array.isArray(msg.sources) && msg.sources.length > 0 && (
          <details className="sources">
            <summary>Sources ({msg.sources.length})</summary>
            <ul>
              {msg.sources.map((s, i) => (
                <li key={i}>
                  <div className="srcTop">
                    <b>{s.file}</b> • chunk <b>{s.chunk}</b>
                  </div>
                  {s.excerpt ? <div className="srcExcerpt">{s.excerpt}</div> : null}
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}
