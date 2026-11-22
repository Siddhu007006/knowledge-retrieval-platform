const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

export type KnowledgeDocument = {
  document_id: string;
  filename: string;
  file_type: "pdf" | "docx" | "txt" | "csv";
  chunk_count: number;
  size_bytes: number;
  created_at: string;
};

export type Citation = {
  document_id: string;
  filename: string;
  page_number: number | null;
  section: string | null;
  chunk_id: string;
  excerpt: string;
};

export type QueryResult = {
  conversation_id: string | null;
  clarification_required: boolean;
  answer: string;
  citations: Citation[];
  grounding_status: "supported" | "partially_supported" | "unsupported" | "insufficient_context" | "clarification_required";
  retrieval_metadata: {
    candidate_count: number;
    reranked_count: number;
    selected_count: number;
    rewrite_applied: boolean;
    rewritten_query: string | null;
    rewrite_reason: string | null;
    retry_count: number;
  };
};

async function readError(response: Response): Promise<Error> {
  try {
    const body = await response.json();
    return new Error(body?.error?.message ?? `Request failed (${response.status})`);
  } catch {
    return new Error(`Request failed (${response.status})`);
  }
}

export async function getDocuments(): Promise<KnowledgeDocument[]> {
  const response = await fetch(`${API_BASE}/documents`, { cache: "no-store" });
  if (!response.ok) throw await readError(response);
  const body = await response.json();
  return body.documents as KnowledgeDocument[];
}

export async function uploadDocument(file: File): Promise<void> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/documents/upload`, { method: "POST", body: form });
  if (!response.ok) throw await readError(response);
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/documents/${documentId}`, { method: "DELETE" });
  if (!response.ok) throw await readError(response);
}

export async function askKnowledgeBase(
  question: string,
  conversationId: string,
  documentIds?: string[],
): Promise<QueryResult> {
  const response = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, conversation_id: conversationId, ...(documentIds?.length ? { document_ids: documentIds } : {}) }),
  });
  if (!response.ok) throw await readError(response);
  return await response.json() as QueryResult;
}
