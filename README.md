# DraftWorks

## Team Name
DraftWorks

## Team Members
- Add names here

## Track
- Add hackathon track here

## What We Built
DraftWorks is an AI-assisted drawing compliance checker for defense/mechanical engineering workflows.

For the MVP, users upload a drawing PDF and optional context files (CSV/TXT/JSON). The app extracts drawing text, detects core sections, compares against known context updates, and returns issues in a structured format suitable for markup or redline overlays.

## Datasets/APIs Used
- User-provided engineering drawing PDFs (primary input)
- User-provided context files (standards/spec/baseline mappings)
- OCR engine: `rapidocr-onnxruntime` (open source)
- PDF parser: `pymupdf`

## How to Run It
### Prerequisites
- Node.js 22+
- Python 3.11+

### Install
```bash
npm install
npm --prefix frontend install
python3 -m pip install -r backend/requirements.txt
```

### Start (single command)
```bash
npm run dev
```

- Frontend: `http://localhost:5173`
- API: `http://localhost:8000`

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
