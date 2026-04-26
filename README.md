# DraftWorks

## Team Name
DraftWorks

### Team Members
- Sanja Kirova
- Axel Ortega
- Emanuel Perez

## Track
GenAI.mil

## What We Built
DraftWorks is an AI-assisted drawing compliance checker for defense/mechanical engineering workflows.
Project built during the 2026 SCSP Hackathon in Boston.

For the MVP, users upload a drawing PDF and optional context files (CSV/TXT/JSON). The app extracts drawing text, detects core sections, compares against known context updates, and returns issues in a structured format suitable for markup or redline overlays.
The main drawing input now supports PDF and image files (JPG/JPEG/PNG/WEBP) with OCR.

## Datasets/APIs Used
- User-provided engineering drawing PDFs (primary input)
- User-provided context files (standards/spec/baseline mappings)
- OCR engine: `rapidocr-onnxruntime` (open source)
- PDF parser: `pymupdf`
- Ollama Cloud API (`https://ollama.com/api/chat`) using `gemma4:31b`
- Ollama local API (`http://localhost:11434/api/chat`) using `gemma4:e4b`
- Render Web Service (hosts FastAPI backend): `https://draftworks-api.onrender.com`
- Vercel (hosts frontend web app): `https://draftworks-app.vercel.app`

## How to Run It
### A) Run Live (fastest)
1. Go to live project URL: [https://draftworks-app.vercel.app](https://draftworks-app.vercel.app)
2. In the web app, select `Run online`
3. By default, use `Use app key` (server-managed key on Render). No key needed for normal use.
4. Optional: switch to `Bring your own key (BYOK)` and paste your own Ollama key.
5. Upload drawing and context files, then click `Compare`
6. Sample documents: `<add Google Drive link here>`

### B) Run Locally + Ollama Cloud Key
1. Install dependencies:
```bash
npm install
npm --prefix frontend install
python3 -m pip install -r backend/requirements.txt
```
2. Start app:
```bash
npm run dev
```
3. Open `http://localhost:5173`
4. Get an Ollama API key:
   1. Open [https://ollama.com/settings/keys](https://ollama.com/settings/keys)
   2. Sign in to your Ollama account
   3. Click `Create key`
   4. Copy the generated key
5. Select `Run online`
6. Choose `Bring your own key (BYOK)` and paste the key
7. Upload drawing/context and click `Compare`

### C) Run Locally + Ollama on Device
1. Install dependencies:
```bash
npm install
npm --prefix frontend install
python3 -m pip install -r backend/requirements.txt
```
2. Start local Ollama and pull model:
```bash
ollama pull gemma4:e4b
```
3. Start app:
```bash
npm run dev
```
4. Open `http://localhost:5173`
5. Select `Run locally on device`
6. Upload drawing/context and run analysis (no Ollama Cloud key required)

### Hosted frontend API config (for Vercel)
The frontend now reads `VITE_API_BASE_URL`.

Use local default:
```bash
VITE_API_BASE_URL=http://localhost:8000
```

For hosted usage, set `VITE_API_BASE_URL` in Vercel project settings to your deployed backend URL.
Reference file: `frontend/.env.example`

### Deploy backend on Render
This repo includes a Render Blueprint at `render.yaml`.

1. In Render, click `New +` -> `Blueprint`.
2. Connect this GitHub repo.
3. Render will detect `render.yaml` and create `draftworks-api`.
4. After deploy, copy your backend URL, for example: `https://draftworks-api.onrender.com`.
5. In Vercel (`draftworks-app` project), set:
   - `VITE_API_BASE_URL=https://YOUR-RENDER-URL`
6. Redeploy frontend on Vercel.

Required Render env vars for cloud mode:
- `OLLAMA_ENABLED=true`
- `OLLAMA_CLOUD_URL=https://ollama.com/api/chat`
- `OLLAMA_CLOUD_MODEL=gemma4:31b`
- `OLLAMA_TIMEOUT_SECONDS=45`

### Optional backend env vars
- `OLLAMA_LOCAL_URL` (default: `http://localhost:11434/api/chat`)
- `OLLAMA_CLOUD_URL` (default: `https://ollama.com/api/chat`)
- `OLLAMA_LOCAL_MODEL` (default: `gemma4:e4b`)
- `OLLAMA_CLOUD_MODEL` (default: `gemma4:31b`)
- `OLLAMA_ENABLED` (default: `true`)
- `OLLAMA_TIMEOUT_SECONDS` (default: `45`)
- `OLLAMA_API_KEY` (server-side key used by default in online mode; users can still override via BYOK)
- `SURREAL_URL` (example: `http://localhost:8001` or your Surreal Cloud endpoint)
- `SURREAL_NS` (namespace)
- `SURREAL_DB` (database)
- `SURREAL_USER` / `SURREAL_PASS` (optional basic auth)
- `SURREAL_TOKEN` (optional bearer token auth)
- `SURREAL_TABLE` (default: `foundational_context`)
- `SURREAL_CHUNK_TABLE` (default: `<SURREAL_TABLE>_chunks`)
- `RAG_CHUNK_SIZE` (default: `1400`)
- `RAG_CHUNK_OVERLAP` (default: `240`)
- `RAG_MAX_DOCS_SCAN` (default: `250`)
- `RAG_MAX_CHUNKS_SCAN` (default: `1200`)

### SurrealDB RAG (Foundational Context)
When `Include foundational org context` is enabled in the app, `/api/analyze` retrieves relevant foundational docs from SurrealDB and merges extracted rules into comparison.

RAG architecture (current):
- Ingestion stores each uploaded foundational document in `SURREAL_TABLE`.
- Documents are chunked and stored in `SURREAL_CHUNK_TABLE` with extracted terms.
- Retrieval ranks chunks by query-term overlap and aggregates top chunks into doc context.
- If chunk data is unavailable, retrieval falls back to doc-level matching.

Ingest foundational context docs:
```bash
curl -X POST "https://draftworks-api.onrender.com/api/foundational-context/upload" \
  -F "files=@/path/to/standards.csv" \
  -F "files=@/path/to/spec_updates.txt"
```

Search stored foundational docs:
```bash
curl "https://draftworks-api.onrender.com/api/foundational-context/search?q=ASTM%20A36"
```

## Output Schema (MVP)
Each detected issue returns:

- `id`: unique issue id
- `issue_type`: `outdated_standard | outdated_spec | material_mismatch | bom_mismatch | missing_reference | ocr_uncertain`
- `severity`: `low | medium | high | critical`
- `message`: short explanation
- `page`: page number when known
- `section`: inferred section (`notes`, `revision_block`, `title_block`, `bom`, `drawing_views`)
- `evidence`: source snippet/token found
- `expected_value`: what should be present
- `found_value`: what was found
- `recommendation`: action to resolve

## MVP Notes
- Drawing-view-to-BOM validation is called out in the UI and implemented as a first-pass check based on extracted item callouts.
- Foundational org context checkbox is wired to SurrealDB-backed retrieval for MVP RAG.

## Analysis Flow
1. OCR/parser extracts drawing text.
2. The backend applies a layout template and splits each page into zones:
   - `notes` (top-left)
   - `revision_block` (top-right)
   - `title_block` (bottom-right)
   - `drawing_area` (center/main viewport)
3. Zone text is extracted from PDF geometry (`fitz` text blocks) and merged per zone.
4. Rule checks run (standards/spec/BOM logic) on extracted page/zone text.
5. If enabled, Surreal RAG retrieves relevant foundational context and merges context rules.
6. Backend builds a prompt with drawing text + zone text + context rules.
7. It calls Gemma via Ollama:
   - Cloud: `gemma4:31b` (online mode)
   - Local: `gemma4:e4b` (local mode)
8. Gemma returns structured issue candidates, then backend normalizes/dedupes and returns final results.
