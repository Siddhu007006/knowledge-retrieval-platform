# M1.1 — Research and Technical Understanding

## 1. Retrieval-Augmented Generation (RAG)

RAG combines information retrieval with language-model generation. The model is not expected to answer solely from its learned parameters: the application first retrieves relevant passages from a controlled knowledge base, then supplies those passages as evidence to the model.

### Indexing path

1. Accept a knowledge-base file.
2. Extract its textual content while retaining useful provenance such as filename, page, section, table, or CSV row.
3. Normalize whitespace and Unicode, then split the text into overlapping chunks.
4. Generate one embedding vector per chunk using the embedding model's document-retrieval task mode.
5. Store vectors in FAISS and store chunk/document metadata alongside them.

### Query path

1. Understand the current question and, where available, its conversation context.
2. Embed the query using the query-retrieval task mode.
3. Run vector similarity search over the knowledge base.
4. Rerank and filter candidates to form a bounded evidence context.
5. Generate a response constrained to that evidence.
6. Verify answer grounding and return source citations. When the evidence is insufficient, abstain rather than inventing an answer.

## 2. Embeddings and semantic similarity

An embedding maps a text passage to a dense numeric vector. Semantically related texts should occupy nearby regions in the embedding space, allowing retrieval even when a user does not repeat the exact words used in a source document.

This implementation uses Gemini embeddings with separate task modes:

- `RETRIEVAL_DOCUMENT` for indexed document chunks.
- `RETRIEVAL_QUERY` for user queries.

FAISS uses an inner-product index over L2-normalized vectors. For normalized vectors, inner product corresponds to cosine similarity. The retrieved score is a semantic similarity signal, not a calibrated probability that the passage is correct.

## 3. Chunking strategy

The initial configuration uses a target of 700 estimated tokens and an overlap of 100 estimated tokens. The current tokenizer-independent estimator uses approximately four characters per token. Paragraph boundaries are preserved where practical, and long paragraphs are split into smaller units. Page and section metadata are carried into each chunk for citation construction.

Why chunking is necessary:

- Whole-document embeddings are too coarse for locating precise supporting passages.
- Very small chunks can lose context required to interpret a statement.
- Overlap reduces the chance that a fact split at a chunk boundary becomes unretrievable.

The chosen values are initial defaults, not universal optimal values. They should be tuned against retrieval evaluation results and the actual structure of the knowledge base.

## 4. Vector search and retrieval quality

Semantic search produces a candidate set. The retrieval pipeline then uses a second scoring pass to order candidates and a context filter to select a small evidence set for generation. The current reranker is an embedding-similarity refinement; it is not a trained cross-encoder reranker.

Retrieval quality is evaluated on two sample domains (HR policy and IT Support). The labelled test set contains:

- Factual queries.
- Procedural queries.
- Comparative queries.
- Unavailable-information queries.

For answerable queries, Top-K retrieval accuracy is measured at chunk level with Hit@K: a query is a hit if one or more labelled relevant chunks appear in the top K results. Each answerable query identifies one or more evidence phrases in its sample source document; the evaluation runner maps those phrases to chunk IDs before scoring. Report Hit@1, Hit@3 and Hit@5. Unanswerable queries are kept in the evaluation set to validate the abstention path, but are not included in the answerable-query retrieval hit-rate denominator.

## 5. Multi-agent query resolution

The term *agent* here means a clearly separated application responsibility with an explicit input/output contract. The milestone does not require five independent LLM processes or unconstrained autonomous agents. LangGraph coordinates the five responsibilities in one bounded workflow.

| Agent | Responsibility | Main inputs | Main outputs |
|---|---|---|---|
| Query Understanding Agent | Interpret the question, use recent context to resolve follow-ups, and refine the search query | Question, recent conversation, document scope | Retrieval-ready query and rewrite reason |
| Retrieval Agent | Retrieve, rerank, and filter evidence chunks | Query, vector index, document scope | Ranked candidates and context package |
| Response Generation Agent | Generate an answer using only selected evidence and validate grounding | Original question, evidence context | Answer and grounding result |
| Clarification Agent | Detect unresolved references when no conversation antecedent exists | Question, conversation history | Clarification decision and follow-up prompt |
| Conversation Memory Agent | Load and persist bounded session-specific messages | Conversation ID, messages | Recent conversation history |

### Orchestration approach

LangGraph is used because the query flow has explicit stages and conditional paths. It supports a clarification branch, a normal retrieval/generation branch, and bounded retry loops when context or grounding is insufficient. The five agent responsibilities remain independently testable modules.

## 6. Web Speech API

The frontend integrates browser speech capabilities rather than sending audio to the Python backend.

- **Speech-to-text:** `SpeechRecognition` or the browser-prefixed `webkitSpeechRecognition` converts microphone speech into a text transcript. The transcript is placed in the normal query input; it then follows the same API and RAG pipeline as typed input.
- **Text-to-speech:** `window.speechSynthesis` and `SpeechSynthesisUtterance` read the generated answer aloud.
- **Fallback:** If the browser does not expose speech recognition or synthesis, typed queries and visible text answers remain available.
- **Permissions:** The browser prompts the user for microphone access. Speech recognition availability and behavior depend on browser implementation and configuration.

## 7. Technology decisions

| Concern | Selection | Reason for M1 |
|---|---|---|
| API/backend | FastAPI + Pydantic | Typed HTTP endpoints and request/response schemas |
| Orchestration | LangGraph | Explicit state, agent stages, branching and bounded retries |
| Foundation model | Google Gemini | Used for query refinement, answer generation and grounding checks |
| Embeddings | Gemini Embedding API | One embedding family with retrieval-specific task modes |
| Vector store | FAISS | Local vector indexing and semantic similarity search without an external service |
| Document extraction | PyMuPDF, python-docx, stdlib CSV | Direct support for the four required upload formats |
| Metadata/chunks | JSON file store | Small local M1 knowledge base without an extra database service |
| Conversation memory | SQLite | Persistent session messages using Python standard library |
| Frontend | Next.js + React + TypeScript | Document management, query interface and typed API integration |
| Speech | Browser Web Speech API | Browser-side input and spoken response with graceful fallback |

## 8. Known limitations

- The chunk token count is estimated by character length, not an exact model tokenizer.
- FAISS and JSON storage are local single-instance persistence; this is not a distributed production vector database.
- Scanned PDFs without embedded text require OCR, which is not included in this milestone.
- CSV is represented as column-labelled row text; complex spreadsheets with multiple sheets are outside the CSV format.
- Browser speech recognition is not standardized uniformly across all browsers and can require network connectivity depending on the browser.
- The current reranking method uses embedding similarity rather than a dedicated cross-encoder model.
