"use client";

import { ChangeEvent, FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { askKnowledgeBase, deleteDocument, getDocuments, KnowledgeDocument, QueryResult, uploadDocument } from "@/lib/api";

type SpeechResult = { 0: { transcript: string }; isFinal: boolean };
type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: { results: ArrayLike<SpeechResult> }) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};
type SpeechWindow = Window & {
  SpeechRecognition?: new () => SpeechRecognitionLike;
  webkitSpeechRecognition?: new () => SpeechRecognitionLike;
};

type Turn = { question: string; result: QueryResult };
const SUPPORTED_FORMATS = ".pdf,.docx,.txt,.csv";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function groundingLabel(status: QueryResult["grounding_status"]): string {
  const labels: Record<QueryResult["grounding_status"], string> = {
    supported: "Evidence supported",
    partially_supported: "Partially supported",
    unsupported: "Not fully verified",
    insufficient_context: "Insufficient evidence",
    clarification_required: "Clarification needed",
  };
  return labels[status];
}

export default function HomePage() {
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loadingDocuments, setLoadingDocuments] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [listening, setListening] = useState(false);
  const [conversationId, setConversationId] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const recognition = useRef<SpeechRecognitionLike | null>(null);

  const refreshDocuments = useCallback(async () => {
    setLoadingDocuments(true);
    try {
      const items = await getDocuments();
      setDocuments(items);
      setSelectedIds((current) => current.filter((id) => items.some((doc) => doc.document_id === id)));
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the knowledge base.");
    } finally {
      setLoadingDocuments(false);
    }
  }, []);

  useEffect(() => {
    setConversationId(window.crypto.randomUUID());
    void refreshDocuments();
  }, [refreshDocuments]);

  async function handleFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) return;
    setUploading(true);
    setError("");
    try {
      for (const file of files) await uploadDocument(file);
      setNotice(`${files.length} document${files.length === 1 ? "" : "s"} indexed successfully.`);
      await refreshDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Document upload failed.");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function removeDocument(documentId: string) {
    try {
      await deleteDocument(documentId);
      setDocuments((current) => current.filter((doc) => doc.document_id !== documentId));
      setSelectedIds((current) => current.filter((id) => id !== documentId));
      setNotice("Document and its indexed chunks were deleted.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete the document.");
    }
  }

  async function submitQuestion(event?: FormEvent) {
    event?.preventDefault();
    const prompt = question.trim();
    if (!prompt || asking) return;
    setAsking(true);
    setError("");
    setNotice("");
    const session = conversationId || window.crypto.randomUUID();
    setConversationId(session);
    try {
      const result = await askKnowledgeBase(prompt, session, selectedIds);
      if (result.conversation_id) setConversationId(result.conversation_id);
      setTurns((current) => [...current, { question: prompt, result }]);
      setQuestion("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "The query could not be resolved.");
    } finally {
      setAsking(false);
    }
  }

  function startVoiceInput() {
    const speechWindow = window as SpeechWindow;
    const SpeechRecognitionConstructor = speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;
    if (!SpeechRecognitionConstructor) {
      setError("Speech recognition is not supported in this browser. Try a recent version of Chrome or Edge.");
      return;
    }
    const instance = new SpeechRecognitionConstructor();
    instance.lang = "en-IN";
    instance.continuous = false;
    instance.interimResults = false;
    instance.onresult = (event) => {
      const transcript = Array.from(event.results).map((item) => item[0].transcript).join(" ").trim();
      if (transcript) setQuestion((current) => current ? `${current} ${transcript}` : transcript);
    };
    instance.onerror = (event) => setError(`Speech recognition error: ${event.error}`);
    instance.onend = () => setListening(false);
    recognition.current = instance;
    setListening(true);
    instance.start();
  }

  function stopVoiceInput() {
    recognition.current?.stop();
    setListening(false);
  }

  function speakAnswer(text: string) {
    if (!("speechSynthesis" in window)) {
      setError("Text-to-speech is not supported in this browser.");
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-IN";
    window.speechSynthesis.speak(utterance);
  }

  function startNewConversation() {
    window.speechSynthesis?.cancel();
    setConversationId(window.crypto.randomUUID());
    setTurns([]);
    setQuestion("");
    setNotice("Started a new conversation. Previous session memory is retained separately.");
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark">KR</div>
        <div className="brand-copy"><strong>Knowledge Retrieval Platform</strong><span>Multi-agent RAG workspace</span></div>
        <div className="topbar-status"><span className="status-dot" /> Knowledge base workspace</div>
      </header>

      <section className="intro">
        <p className="eyebrow">MILESTONE 1 · FOUNDATION & KNOWLEDGE RETRIEVAL</p>
        <h1>Ask your knowledge base.</h1>
        <p>Upload source documents, retrieve relevant evidence, and receive grounded answers with citations.</p>
      </section>

      <div className="workspace-grid">
        <aside className="panel knowledge-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">KNOWLEDGE BASE</p><h2>Documents</h2></div>
            <span className="count-chip">{documents.length}</span>
          </div>
          <div className="upload-box">
            <div className="upload-symbol">↑</div>
            <strong>{uploading ? "Indexing documents…" : "Add knowledge sources"}</strong>
            <span>PDF, DOCX, TXT and CSV · up to 20 MB each</span>
            <button className="button button-primary" onClick={() => fileInput.current?.click()} disabled={uploading}>Choose files</button>
            <input ref={fileInput} type="file" accept={SUPPORTED_FORMATS} multiple hidden onChange={handleFiles} />
          </div>
          <div className="document-toolbar">
            <span>Indexed sources</span>
            <button className="text-button" onClick={() => void refreshDocuments()}>Refresh</button>
          </div>
          <div className="document-list">
            {loadingDocuments ? <p className="muted empty-state">Loading knowledge base…</p> : documents.length === 0 ? (
              <p className="muted empty-state">No documents yet. Upload your first source document.</p>
            ) : documents.map((doc) => (
              <div className="document-row" key={doc.document_id}>
                <input aria-label={`Select ${doc.filename}`} type="checkbox" checked={selectedIds.includes(doc.document_id)} onChange={(event) => setSelectedIds((current) => event.target.checked ? [...current, doc.document_id] : current.filter((id) => id !== doc.document_id))} />
                <div className="file-type">{doc.file_type.toUpperCase()}</div>
                <div className="document-info"><strong title={doc.filename}>{doc.filename}</strong><span>{doc.chunk_count} chunks · {formatBytes(doc.size_bytes)}</span></div>
                <button className="icon-button" title="Delete document" aria-label={`Delete ${doc.filename}`} onClick={() => void removeDocument(doc.document_id)}>×</button>
              </div>
            ))}
          </div>
          <p className="selection-note">{selectedIds.length ? `${selectedIds.length} selected · retrieval is scoped to these documents` : "No selection · retrieval searches all indexed documents"}</p>
        </aside>

        <section className="panel query-panel">
          <div className="panel-heading query-heading">
            <div><p className="eyebrow">QUERY RESOLUTION</p><h2>Knowledge assistant</h2></div>
            <button className="text-button" onClick={startNewConversation}>New conversation ↗</button>
          </div>
          <div className="conversation" aria-live="polite">
            {turns.length === 0 ? (
              <div className="welcome-state"><div className="welcome-icon">⌕</div><h3>What do you need to know?</h3><p>Ask a question about your uploaded documents. The system will retrieve evidence and cite the source chunks used.</p><div className="example-queries"><button onClick={() => setQuestion("Summarize the key points in the uploaded documents.")}>Summarize my documents</button><button onClick={() => setQuestion("What is the process described in the policy?")}>Find a procedure</button></div></div>
            ) : turns.map((turn, index) => (
              <article className="turn" key={`${index}-${turn.result.conversation_id ?? "session"}`}>
                <div className="user-question"><span className="avatar user-avatar">Y</span><div><p className="message-label">YOU</p><p>{turn.question}</p></div></div>
                <div className="assistant-answer">
                  <span className="avatar assistant-avatar">AI</span>
                  <div className="answer-content">
                    <div className="answer-topline"><span className={`grounding-tag ${turn.result.grounding_status}`}>{groundingLabel(turn.result.grounding_status)}</span><button className="text-button" onClick={() => speakAnswer(turn.result.answer)}>Speak answer</button></div>
                    <p className="answer-text">{turn.result.answer}</p>
                    {turn.result.citations.length > 0 && <div className="citations"><p className="message-label">RETRIEVED SOURCES</p>{turn.result.citations.map((source) => <div className="citation-card" key={source.chunk_id}><strong>{source.filename}{source.page_number ? ` · page ${source.page_number}` : ""}{source.section ? ` · ${source.section}` : ""}</strong><p>{source.excerpt}</p><small>Chunk {source.chunk_id.slice(0, 8)}</small></div>)}</div>}
                    <div className="retrieval-stats"><span>{turn.result.retrieval_metadata.candidate_count} candidates</span><span>{turn.result.retrieval_metadata.reranked_count} reranked</span><span>{turn.result.retrieval_metadata.selected_count} context chunks</span></div>
                  </div>
                </div>
              </article>
            ))}
            {asking && <div className="thinking"><span className="loader" /> Understanding query, retrieving evidence and generating response…</div>}
          </div>
          <form className="composer" onSubmit={submitQuestion}>
            <label className="sr-only" htmlFor="question">Ask a question</label>
            <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submitQuestion(); } }} placeholder="Ask a question about your knowledge base…" rows={3} />
            <div className="composer-footer"><div className="composer-hint">Enter to send · Shift + Enter for a new line</div><div className="composer-actions"><button type="button" className={`button button-secondary ${listening ? "listening" : ""}`} onClick={listening ? stopVoiceInput : startVoiceInput}>{listening ? "Stop listening" : "Voice input"}</button><button className="button button-primary" type="submit" disabled={asking || !question.trim()}>{asking ? "Resolving…" : "Ask knowledge base →"}</button></div></div>
          </form>
        </section>
      </div>

      {error && <div className="notice notice-error" role="alert">{error}<button onClick={() => setError("")}>×</button></div>}
      {notice && <div className="notice notice-success" role="status">{notice}<button onClick={() => setNotice("")}>×</button></div>}
      <footer className="footer"><span>Milestone 1 · RAG ingestion · Multi-agent query resolution · Retrieval evaluation</span><span>Speech recognition and synthesis use browser Web Speech APIs.</span></footer>
    </main>
  );
}
