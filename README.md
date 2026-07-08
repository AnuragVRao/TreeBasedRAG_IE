# EywaAI | Tree-Based RAG

A Retrieval-Augmented Generation (RAG) system built around the **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) approach. Instead of retrieving flat, fixed-size chunks, EywaAI recursively clusters and summarizes a document into a multi-layer tree — leaf nodes hold raw text chunks, and each layer above holds LLM-generated summaries of the layer below. This lets the same index answer both narrow factual questions (via leaf chunks) and broad thematic questions (via high-level summaries).

## 🚀 Features

- **RAPTOR Tree Construction**: Recursively clusters chunk embeddings (UMAP + Gaussian Mixture Models, cluster count chosen via BIC) and summarizes each cluster with a local LLM (via Ollama) until a single root layer remains.
- **Hybrid Querying**: A zero-shot NLI query classifier (with a fast regex layer for clear-cut cases) labels each query as `broad`, `specific`, `definitional`, or `comparative`, then routes it to the matching retrieval strategy — flat "collapsed-tree" kNN for broad queries, top-down layer-by-layer traversal for specific ones.
- **Comparative Query Decomposition**: "X vs Y" / "difference between X and Y" queries are split into two sub-queries, retrieved independently, and merged — avoiding the asymmetric-branch problem where a single traversal only follows the higher-scoring concept.
- **Out-of-Scope Detection**: If the best-matching node's similarity score falls below a threshold, the system reports that the document doesn't cover the topic instead of hallucinating an answer.
- **MMR Deduplication**: Post-retrieval, near-duplicate nodes (cosine similarity above a threshold) are dropped so the context window isn't wasted on redundant text.
- **Optional Web Search Fallback**: If in-tree retrieval is weak (or empty), DuckDuckGo search snippets are appended to the context.
- **Faithfulness Verification** (optional, off by default): An LLM fact-checks its own cluster summaries against the source text and retries generation if it detects unsupported claims.
- **Persistent Storage & Resumable Builds**: Trees are saved to disk (`my_tree/`) and reloaded on restart; per-layer summarization checkpoints let a crashed/interrupted build resume without recomputing already-summarized clusters.
- **Content-Hash Deduplication**: Uploaded files are hashed (SHA-256) so re-uploading the same PDF reuses the existing tree instead of rebuilding it.
- **React + TypeScript Frontend**: Chat-style UI with a document sidebar, and modals for inspecting the tree structure and the exact nodes retrieved for an answer.

---

## 🐳 Quick Start (Docker - Recommended)

The easiest way to run the system is using Docker Compose. This handles all dependencies, including the spaCy language models and environment configurations.

### 1. Prerequisites
- Docker & Docker Desktop installed.
- **Ollama** running on your host machine (Mac/Windows).

### 2. Launch
```bash
# Navigate to project root
cd TreeBasedRAG_IE

# Build and start in detached mode
docker compose up --build -d
```

### 3. Access
* **Frontend**: [http://localhost:5173](http://localhost:5173)
* **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

> [!TIP]
> The system automatically bridges to your host machine's Ollama instance via `host.docker.internal`. Ensure Ollama is running and accessible.

---

## 🛠️ Manual Development Setup

If you prefer to run the services locally without Docker:

### Backend (FastAPI)
1. **Python 3.9+** is required.
2. **Install Dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   # Download the required spaCy model
   python -m spacy download en_core_web_sm
   ```
3. **Run Server**:
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python3 -m uvicorn backend.main:app --reload
   ```

### Frontend (React + Vite)
1. **Node.js v18+** is required.
2. **Install & Run**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Available at `http://localhost:5173`.

---

## 🛠️ Optimizing Ollama Concurrency

To speed up the RAPTOR tree-building process, set the `OLLAMA_NUM_PARALLEL` environment variable **before** starting your Ollama server. This allows concurrent summarization of document clusters.

- **8GB - 12GB RAM**: `OLLAMA_NUM_PARALLEL=2`
- **16GB - 24GB RAM**: `OLLAMA_NUM_PARALLEL=4`
- **32GB+ RAM**: `OLLAMA_NUM_PARALLEL=6`

**Mac Users**: 
Run `launchctl setenv OLLAMA_NUM_PARALLEL 2` and restart the Ollama app.

---

## 📂 Project Structure

```
pipeline.py                 # RaptorPipeline — orchestrates build + query end to end
main.py                     # CLI entry point for local testing (no backend/frontend needed)

ingestion/pdf_parser.py     # PyMuPDF text extraction, section-aware chunking
embedding/embedder.py       # SBERT/BGE sentence embeddings (L2-normalized)
tree/
  clustering.py             # UMAP dimensionality reduction + GMM soft clustering (BIC-selected k)
  summarization.py          # LLM cluster summarization + faithfulness verification prompts/client
  tree_builder.py           # Layer-by-layer tree construction loop, async summarization, checkpointing
  tree_serializer.py        # Save/load tree to/from disk (tree.json + embeddings.npz)
  node.py                   # RaptorNode / RaptorTree dataclasses
retrieval/
  query_classifier.py       # Zero-shot NLI + regex query-type classifier
  retriever.py               # Collapsed-tree vs traversal retrieval, MMR dedup, out-of-scope check
  context_assembler.py       # Assembles retrieved nodes into a token-budgeted context string
  web_searcher.py            # DuckDuckGo web-search fallback
generation/generator.py     # Final answer generation (factual vs inferential prompting)

backend/                    # FastAPI application
  api/                      # Route handlers (documents, query, tree, conversations, analytics)
  services/                 # Business logic gluing RaptorPipeline to the API layer
  schemas/                  # Pydantic request/response models
  core/config.py            # Environment-driven settings

frontend/                   # React + TypeScript (Vite) chat UI
  src/app/                  # App shell, Sidebar, QuerySection, Tree/Nodes modals
  src/lib/api.ts            # Typed fetch wrappers for the backend API

data/                       # Uploaded source documents, deduplicated by content hash
my_tree/                    # Serialized RAPTOR trees, one directory per document
```

---

## 🧠 How a Query Flows Through the System

1. **Classify** — `ZeroShotQueryClassifier` labels the query (`specific` / `broad` / `definitional` / `comparative`) using fast regex rules first, falling back to a zero-shot NLI model for ambiguous phrasing.
2. **Decompose (if comparative)** — "A vs B" queries are split into two independent sub-queries so each concept gets its own retrieval pass.
3. **Route** — `broad` queries use collapsed-tree kNN (flat search across every node, budgeted by token count); everything else uses top-down traversal from the root layer, following the top-k highest-scoring branches down to the leaves.
4. **Gate** — if the best similarity score is below `relevance_threshold`, the query is treated as out-of-scope and short-circuited.
5. **Deduplicate** — MMR drops nodes that are near-duplicates of higher-ranked nodes already selected.
6. **Assemble** — nodes are ordered leaf-first into a single context string within the token budget, so truncation drops summary nodes before source chunks.
7. **Augment (optional)** — if the best node score is still weak, web search snippets are appended.
8. **Generate** — the LLM answers using a factual or inferential system prompt depending on query type, grounded strictly in the assembled context.

---

## 📝 Usage

1. **Upload**: Use the sidebar to upload a PDF. Its content is hashed and saved to `data/`.
2. **Build**: The system automatically builds a RAPTOR tree. This is saved to `my_tree/` and reloaded on future restarts.
3. **Query**: Ask questions! The system will classify your query and retrieve context from the appropriate layers of the tree.
4. **Visualize**: Check the "Tree Structure" and "Retrieved Nodes" panels to see exactly how the RAG engine picked its context.
