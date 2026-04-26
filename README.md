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
- Ollama Cloud API (`https://ollama.com/api/chat`)
- Render Web Service (hosts FastAPI backend): `https://draftworks-api.onrender.com`
- Vercel (hosts frontend web app): `https://draftworks-app.vercel.app`

## How to Run It
### A) Run Live (fastest)
1. Go to live project URL: [https://draftworks-app.vercel.app](https://draftworks-app.vercel.app)
2. Get Ollama API key:
   1. Open [https://ollama.com/settings/keys](https://ollama.com/settings/keys)
   2. Sign in to your Ollama account
   3. Click `Create key`
   4. Copy the generated key
3. In the web app, select `Run online`
4. Paste your key into `Ollama API key`
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
4. Select `Run online`
5. Paste your Ollama API key
6. Upload drawing/context and click `Compare`

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
6. Upload drawing/context and run analysis

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
