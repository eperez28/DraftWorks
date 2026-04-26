# DraftWorks

## Team Name
DraftWorks

### Team Members
- Emanuel
- Sanja
- Axel

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
- Ollama Cloud API (`https://ollama.com/api/chat`)

## How to Run It
Live project URL: [https://draftworks-app.vercel.app](https://draftworks-app.vercel.app)

### Prerequisites
- Node.js 22+
- Python 3.11+

### Install
```bash
npm install
npm --prefix frontend install
python3 -m pip install -r backend/requirements.txt
```

### Start locally (single command)
```bash
npm run dev
```

- Frontend: `http://localhost:5173`
- API: `http://localhost:8000`

### Three ways to run
1. Use our webapp + bring your own Ollama Cloud key
2. Run locally + bring your own Ollama Cloud key (`Run online`)
3. Run locally + run Ollama on-device (`Run locally on device`)

### 1) Use our webapp + bring your own Ollama Cloud key
Web app: [https://draftworks-app.vercel.app](https://draftworks-app.vercel.app)

1. Create an Ollama API key at `https://ollama.com/settings/keys`.
2. Open the webapp and select `Run online (Ollama Cloud)`.
3. Paste the key in `Ollama API key`.
4. Upload drawing/context and run analysis.

### 2) Run locally + bring your own Ollama Cloud key
1. Start app with `npm run dev`.
2. Open `http://localhost:5173`.
3. In webapp select `Run online`.
4. Paste your Ollama API key and run analysis.

### 3) Run locally + run Ollama on-device
1. Start local Ollama and pull model:
```bash
ollama pull gemma4:e4b
```
2. Start app with `npm run dev`.
3. Open `http://localhost:5173`.
4. In webapp select `Run locally on device`.
5. Upload drawing/context and run analysis.

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
- `OLLAMA_API_KEY` (optional server-side fallback for online mode)

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
- Foundational org context checkbox is present in UI; SurrealDB integration can be layered next for persistent references.
