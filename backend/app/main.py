from __future__ import annotations

import io
import json
import os
import re
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal
from urllib import error, request
from xml.etree import ElementTree as ET

import fitz
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None  # type: ignore[assignment]

app = FastAPI(title="DraftWorks API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_engine = RapidOCR() if RapidOCR else None
OLLAMA_LOCAL_URL = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434/api/chat")
OLLAMA_CLOUD_URL = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com/api/chat")
OLLAMA_LOCAL_MODEL = os.getenv("OLLAMA_LOCAL_MODEL", "gemma4:e4b")
OLLAMA_CLOUD_MODEL = os.getenv("OLLAMA_CLOUD_MODEL", "gemma4:31b")
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "true").lower() != "false"
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
SURREAL_URL = os.getenv("SURREAL_URL", "").strip()
SURREAL_NS = os.getenv("SURREAL_NS", "").strip()
SURREAL_DB = os.getenv("SURREAL_DB", "").strip()
SURREAL_USER = os.getenv("SURREAL_USER", "").strip()
SURREAL_PASS = os.getenv("SURREAL_PASS", "").strip()
SURREAL_TOKEN = os.getenv("SURREAL_TOKEN", "").strip()
SURREAL_TABLE = os.getenv("SURREAL_TABLE", "foundational_context").strip() or "foundational_context"
SURREAL_CHUNK_TABLE = os.getenv("SURREAL_CHUNK_TABLE", f"{SURREAL_TABLE}_chunks").strip() or f"{SURREAL_TABLE}_chunks"
RAG_CHUNK_SIZE = max(400, int(os.getenv("RAG_CHUNK_SIZE", "1400")))
RAG_CHUNK_OVERLAP = max(50, int(os.getenv("RAG_CHUNK_OVERLAP", "240")))
RAG_MAX_DOCS_SCAN = max(50, int(os.getenv("RAG_MAX_DOCS_SCAN", "250")))
RAG_MAX_CHUNKS_SCAN = max(100, int(os.getenv("RAG_MAX_CHUNKS_SCAN", "1200")))
MAX_DRAWING_BYTES = max(1_000_000, int(os.getenv("MAX_DRAWING_BYTES", "20000000")))
MAX_PDF_PAGES = max(1, int(os.getenv("MAX_PDF_PAGES", "6")))
OCR_PDF_DPI = max(72, min(300, int(os.getenv("OCR_PDF_DPI", "170"))))
OCR_MAX_IMAGE_SIDE = max(512, int(os.getenv("OCR_MAX_IMAGE_SIDE", "2200")))
OCR_ENABLE_PREPROCESSING = os.getenv("OCR_ENABLE_PREPROCESSING", "true").lower() != "false"
OCR_PREPROCESS_MAX_PIXELS = max(1_000_000, int(os.getenv("OCR_PREPROCESS_MAX_PIXELS", "8000000")))
SURREAL_ZONE_TABLE = os.getenv("SURREAL_ZONE_TABLE", "drawing_zone_items").strip() or "drawing_zone_items"


class ComplianceIssue(BaseModel):
    id: str
    issue_type: Literal[
        "outdated_standard",
        "outdated_spec",
        "material_mismatch",
        "bom_mismatch",
        "missing_reference",
        "ocr_uncertain",
    ]
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    page: int | None
    section: str
    evidence: str
    expected_value: str | None
    found_value: str | None
    recommendation: str


class AnalysisMeta(BaseModel):
    pages_processed: int
    used_foundational_context: bool
    context_files_count: int
    inference_mode: Literal["online", "local"]
    llm_enabled: bool
    llm_used: bool
    llm_model: str | None
    llm_endpoint: str | None
    llm_error: str | None
    foundational_context_error: str | None


class AnalysisResult(BaseModel):
    run_id: str
    summary: str
    sections_detected: list[str]
    issues: list[ComplianceIssue]
    zone_rows: list["ZoneItemRow"]
    comparison_rows: list["ComparisonRow"]
    zone_markdown: str
    meta: AnalysisMeta


class ZoneItemRow(BaseModel):
    page: int
    zone: str
    object_key: str
    object_values: list[str]
    line_number: int


class ComparisonRow(BaseModel):
    sheet: int
    zone: str
    existing_text: str
    replace_with: str
    change_type: str
    source_basis: str
    priority: str
    notes: str
    status: Literal["change", "review_only", "unsure"] = "unsure"


class ZoneExtractResult(BaseModel):
    source_name: str
    pages_processed: int
    row_count: int
    markdown: str
    rows: list[ZoneItemRow]


@dataclass
class ContextRule:
    old_value: str
    new_value: str
    source_name: str | None = None
    source_ref: str | None = None


@dataclass
class ContextEntry:
    object_name: str | None
    key: str
    value: str
    source_name: str | None = None
    source_ref: str | None = None


@dataclass
class LlmRuntimeConfig:
    mode: Literal["online", "local"]
    model: str
    url: str
    api_key: str | None


@dataclass
class FoundationalDoc:
    id: str
    source_name: str
    content: str


@dataclass
class FoundationalChunk:
    id: str
    doc_id: str
    source_name: str
    chunk_index: int
    content: str
    terms: list[str]


SECTION_PATTERNS = {
    "notes": re.compile(r"\bnotes?\b", re.IGNORECASE),
    "title_block": re.compile(r"\btitle\s*block\b|\bdrawing\s*title\b", re.IGNORECASE),
    "revision_block": re.compile(r"\brev(?:ision)?\b|\bchange\s*log\b", re.IGNORECASE),
    "bom": re.compile(r"\bbill\s+of\s+materials\b|\bbom\b|\bparts\s+list\b", re.IGNORECASE),
    "drawing_views": re.compile(r"\bsection\s+[a-z]\b|\bview\b|\bdetail\b", re.IGNORECASE),
}

STANDARD_PATTERN = re.compile(r"\b(?:ASTM|MIL-STD|ISO|SAE)[\s\-]*[A-Z0-9\-.]+\b")
ITEM_CALLOUT_PATTERN = re.compile(r"\b(?:ITEM|ITM|BALLOON)\s*#?\s*(\d{1,3})\b", re.IGNORECASE)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ZONE_ORDER = ["title_block", "revision_block", "notes", "drawing_area"]
ZONE_RECTS = {
    # Normalized coordinates tuned to your sketch/template:
    # (x0, y0, x1, y1), with x/y in [0, 1]
    "notes": (0.0, 0.0, 0.34, 0.34),
    "revision_block": (0.60, 0.0, 1.0, 0.26),
    "title_block": (0.45, 0.75, 1.0, 1.0),
    "drawing_area": (0.06, 0.12, 0.98, 0.92),
}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/foundational-context/upload")
async def upload_foundational_context(
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    ensure_surreal_configured()
    inserted = 0
    skipped: list[str] = []

    for file in files:
        raw = await file.read()
        content = extract_text_for_foundational_doc(raw, file.filename)
        if not content.strip():
            skipped.append(file.filename or "unknown")
            continue

        terms = extract_terms(content, limit=24)
        source_name = file.filename or f"context-{inserted+1}.txt"
        store_foundational_doc(source_name=source_name, content=content, terms=terms)
        inserted += 1

    return {
        "stored": inserted,
        "skipped": skipped,
        "surreal_enabled": True,
    }


@app.get("/api/foundational-context/search")
def search_foundational_context(q: str, limit: int = 5) -> dict[str, Any]:
    ensure_surreal_configured()
    capped_limit = max(1, min(limit, 20))
    docs = retrieve_foundational_docs_for_query(q, limit=capped_limit)
    return {
        "query": q,
        "count": len(docs),
        "results": [
            {
                "id": doc.id,
                "source_name": doc.source_name,
                "snippet": truncate(doc.content, 220),
            }
            for doc in docs
        ],
    }


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(
    drawing: UploadFile = File(...),
    context_files: list[UploadFile] = File(default=[]),
    use_foundational_context: bool = Form(default=False),
    inference_mode: Literal["online", "local"] = Form(default="online"),
    ollama_api_key: str | None = Form(default=None),
) -> AnalysisResult:
    drawing_bytes = await drawing.read()
    if len(drawing_bytes) > MAX_DRAWING_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Drawing file exceeds limit of {MAX_DRAWING_BYTES // 1_000_000} MB.",
        )
    page_texts, ocr_low_conf_pages, page_zone_texts = extract_drawing_text(
        drawing_bytes=drawing_bytes,
        filename=drawing.filename,
        content_type=drawing.content_type,
    )
    zone_rows = build_zone_rows(page_zone_texts)
    zone_markdown = render_zone_rows_markdown(zone_rows, drawing.filename or "drawing")
    sections = detect_sections(page_texts, page_zone_texts)

    context_payloads = await read_uploaded_context_payloads(context_files)
    context_rules = extract_context_rules_from_payloads(context_payloads)
    context_entries = extract_context_entries_from_payloads(context_payloads)
    context_sources = sorted({name for name, _ in context_payloads if name})
    foundational_error: str | None = None
    if use_foundational_context:
        try:
            query_text = " ".join(page_texts[:2])[:6000]
            foundational_docs = retrieve_foundational_docs_for_query(query_text, limit=8)
            context_rules.extend(extract_rules_from_foundational_docs(foundational_docs))
        except HTTPException as exc:
            foundational_error = sanitize_service_error(exc.detail, service="SurrealDB")
        except Exception as exc:
            foundational_error = sanitize_service_error(str(exc), service="SurrealDB")
    llm_runtime = resolve_llm_runtime(inference_mode, ollama_api_key)
    llm_rows, llm_used, llm_error = find_llm_change_rows(
        page_texts=page_texts,
        page_zone_texts=page_zone_texts,
        context_rules=context_rules,
        context_entries=context_entries,
        context_payloads=context_payloads,
        sections=sections,
        llm_runtime=llm_runtime,
        context_sources=context_sources,
    )
    fallback_rows = compare_zone_rows_to_context(zone_rows, context_entries, context_rules)
    comparison_rows = filter_rows_for_display(llm_rows if llm_rows else fallback_rows)
    issues: list[ComplianceIssue] = []

    issues.extend(find_outdated_references(page_texts, context_rules))
    issues.extend(find_bom_view_mismatches(page_texts))

    for page in ocr_low_conf_pages:
        issues.append(
            ComplianceIssue(
                id=f"ocr-{page}",
                issue_type="ocr_uncertain",
                severity="low",
                message="OCR confidence on this page was lower than ideal.",
                page=page,
                section="drawing_views",
                evidence="Text extraction relied on OCR fallback.",
                expected_value=None,
                found_value=None,
                recommendation="Review this page manually during final engineering signoff.",
            )
        )

    summary = f"Processed {len(page_texts)} page(s). Detected {len(issues)} issue(s)."
    return AnalysisResult(
        run_id=str(uuid.uuid4()),
        summary=summary,
        sections_detected=sections,
        issues=issues,
        zone_rows=zone_rows,
        comparison_rows=comparison_rows,
        zone_markdown=zone_markdown,
        meta=AnalysisMeta(
            pages_processed=len(page_texts),
            used_foundational_context=use_foundational_context,
            context_files_count=len(context_files),
            inference_mode=inference_mode,
            llm_enabled=OLLAMA_ENABLED,
            llm_used=llm_used,
            llm_model=llm_runtime.model if OLLAMA_ENABLED else None,
            llm_endpoint=llm_runtime.url if OLLAMA_ENABLED else None,
            llm_error=llm_error,
            foundational_context_error=foundational_error,
        ),
    )


@app.post("/api/zones/extract", response_model=ZoneExtractResult)
async def extract_zones(
    drawing: UploadFile = File(...),
    persist_to_surreal: bool = Form(default=False),
) -> ZoneExtractResult:
    drawing_bytes = await drawing.read()
    if len(drawing_bytes) > MAX_DRAWING_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Drawing file exceeds limit of {MAX_DRAWING_BYTES // 1_000_000} MB.",
        )

    page_texts, _, page_zone_texts = extract_drawing_text(
        drawing_bytes=drawing_bytes,
        filename=drawing.filename,
        content_type=drawing.content_type,
    )
    rows = build_zone_rows(page_zone_texts)
    markdown = render_zone_rows_markdown(rows, source_name=drawing.filename or "drawing")

    if persist_to_surreal:
        ensure_surreal_configured()
        store_zone_rows(source_name=drawing.filename or "drawing", rows=rows)

    return ZoneExtractResult(
        source_name=drawing.filename or "drawing",
        pages_processed=len(page_texts),
        row_count=len(rows),
        markdown=markdown,
        rows=rows,
    )


def extract_drawing_text(
    drawing_bytes: bytes,
    filename: str | None,
    content_type: str | None,
) -> tuple[list[str], list[int], list[dict[str, str]]]:
    lower_name = (filename or "").lower()
    ext = os.path.splitext(lower_name)[1]
    ctype = (content_type or "").lower()

    if ctype == "application/pdf" or ext == ".pdf":
        return extract_pdf_text_with_ocr(drawing_bytes)

    if ctype.startswith("image/") or ext in IMAGE_EXTENSIONS:
        return extract_image_text_with_ocr(drawing_bytes)

    raise HTTPException(
        status_code=400,
        detail="Unsupported drawing file type. Upload PDF, JPG, JPEG, PNG, or WEBP.",
    )


def extract_pdf_text_with_ocr(pdf_bytes: bytes) -> tuple[list[str], list[int], list[dict[str, str]]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_texts: list[str] = []
    low_conf_pages: list[int] = []
    page_zone_texts: list[dict[str, str]] = []

    try:
        for i, page in enumerate(doc):
            if i >= MAX_PDF_PAGES:
                break
            text = page.get_text("text") or ""
            zone_text = extract_pdf_zone_text(page)
            page_zone_texts.append(zone_text)

            if len(text.strip()) >= 100:
                page_texts.append(text)
                continue

            if ocr_engine is None:
                page_texts.append(text)
                low_conf_pages.append(i + 1)
                continue

            pix = page.get_pixmap(dpi=OCR_PDF_DPI, colorspace=fitz.csGRAY, alpha=False)
            ocr_lines, low_conf = run_ocr_with_preprocessing(pix.tobytes("png"))

            if not ocr_lines:
                page_texts.append(text.strip())
                low_conf_pages.append(i + 1)
                continue

            merged = text.strip()
            ocr_text = "\n".join(ocr_lines)
            if merged and ocr_text:
                merged = f"{merged}\n{ocr_text}"
            elif ocr_text:
                merged = ocr_text

            page_texts.append(merged)
            if low_conf:
                low_conf_pages.append(i + 1)
    finally:
        doc.close()

    return page_texts, low_conf_pages, page_zone_texts


def extract_image_text_with_ocr(image_bytes: bytes) -> tuple[list[str], list[int], list[dict[str, str]]]:
    if ocr_engine is None:
        return [""], [1], [{"drawing_area": ""}]

    lines, low_conf = run_ocr_with_preprocessing(image_bytes)
    extracted = "\n".join(lines).strip()
    if not extracted:
        return [""], [1], [{"drawing_area": ""}]
    if low_conf:
        return [extracted], [1], [{"drawing_area": extracted}]
    return [extracted], [], [{"drawing_area": extracted}]


def extract_pdf_zone_text(page: fitz.Page) -> dict[str, str]:
    blocks = page.get_text("blocks")
    width = max(1.0, float(page.rect.width))
    height = max(1.0, float(page.rect.height))
    bucket: dict[str, list[str]] = {zone: [] for zone in ZONE_ORDER}

    for block in blocks:
        if not isinstance(block, (list, tuple)) or len(block) < 5:
            continue
        text = str(block[4]).strip()
        if not text:
            continue
        try:
            x0 = float(block[0])
            y0 = float(block[1])
            x1 = float(block[2])
            y1 = float(block[3])
        except (TypeError, ValueError):
            continue

        cx = ((x0 + x1) / 2.0) / width
        cy = ((y0 + y1) / 2.0) / height
        zone_name = classify_zone(cx, cy)
        if not zone_name:
            continue
        bucket[zone_name].append(text)

    result: dict[str, str] = {}
    for zone_name in ZONE_ORDER:
        merged = "\n".join(bucket[zone_name]).strip()
        if merged:
            result[zone_name] = merged
    return result


def classify_zone(cx: float, cy: float) -> str | None:
    for zone_name in ZONE_ORDER:
        x0, y0, x1, y1 = ZONE_RECTS[zone_name]
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return zone_name
    return None


def build_zone_rows(page_zone_texts: list[dict[str, str]]) -> list[ZoneItemRow]:
    rows: list[ZoneItemRow] = []
    for page_idx, zone_map in enumerate(page_zone_texts, start=1):
        for zone_name in ZONE_ORDER:
            zone_text = zone_map.get(zone_name, "")
            if not zone_text.strip():
                continue
            zone_rows = parse_zone_rows(page_idx=page_idx, zone=zone_name, text=zone_text)
            rows.extend(zone_rows)
    return rows


def parse_zone_rows(page_idx: int, zone: str, text: str) -> list[ZoneItemRow]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows: list[ZoneItemRow] = []

    for line_idx, line in enumerate(lines, start=1):
        if zone == "notes":
            rows.append(parse_note_row(page_idx, zone, line_idx, line))
            continue

        key, value = split_key_value(line)
        if not key:
            key = f"{zone}_item_{line_idx}"
        if not value:
            value = line
        values = split_multi_values(value)
        rows.append(
            ZoneItemRow(
                page=page_idx,
                zone=zone,
                object_key=normalize_key(key),
                object_values=values,
                line_number=line_idx,
            )
        )

    return rows


def parse_note_row(page_idx: int, zone: str, line_idx: int, line: str) -> ZoneItemRow:
    match = re.match(r"^(\d+)[\).\s-]+(.+)$", line)
    if match:
        key = f"note_{match.group(1)}"
        value = match.group(2).strip()
    else:
        key = f"note_{line_idx}"
        value = line
    return ZoneItemRow(
        page=page_idx,
        zone=zone,
        object_key=key,
        object_values=split_multi_values(value),
        line_number=line_idx,
    )


def split_key_value(line: str) -> tuple[str, str]:
    if ":" in line:
        left, right = line.split(":", maxsplit=1)
        if left.strip() and right.strip():
            return left.strip(), right.strip()

    spaced = re.split(r"\s{2,}", line)
    if len(spaced) >= 2:
        return spaced[0].strip(), " | ".join(part.strip() for part in spaced[1:] if part.strip())

    return "", line.strip()


def split_multi_values(value: str) -> list[str]:
    if not value.strip():
        return []
    parts = re.split(r"\s\|\s|;|,\s{2,}", value)
    cleaned = [part.strip() for part in parts if part.strip()]
    return cleaned or [value.strip()]


def normalize_key(value: str) -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_") or "item"


def render_zone_rows_markdown(rows: list[ZoneItemRow], source_name: str) -> str:
    lines = [f"# Zone Extraction: {source_name}", "", "| page | zone | object_key | object_values |", "|---|---|---|---|"]
    for row in rows:
        values = "<br>".join(row.object_values) if row.object_values else ""
        lines.append(f"| {row.page} | {row.zone} | {row.object_key} | {values} |")
    return "\n".join(lines)


def run_ocr_with_preprocessing(image_bytes: bytes) -> tuple[list[str], bool]:
    base_image = normalize_image_for_ocr(image_bytes)
    candidates = [base_image]

    if OCR_ENABLE_PREPROCESSING and cv2 is not None and np is not None:
        arr = np.frombuffer(base_image, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is not None:
            height, width = decoded.shape[:2]
            if (height * width) > OCR_PREPROCESS_MAX_PIXELS:
                decoded = resize_image_max_side(decoded, OCR_MAX_IMAGE_SIDE)
            gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
            denoised = cv2.bilateralFilter(gray, 7, 35, 35)
            adaptive = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                10,
            )
            enlarged = cv2.resize(
                adaptive,
                None,
                fx=1.6,
                fy=1.6,
                interpolation=cv2.INTER_CUBIC,
            )
            ok, preprocessed = cv2.imencode(".png", enlarged)
            if ok:
                candidates.append(preprocessed.tobytes())

    best_lines: list[str] = []
    best_quality = -1.0
    best_low_conf = True

    for candidate in candidates:
        ocr_result, _ = ocr_engine(candidate)
        if not ocr_result:
            continue

        lines, avg_conf = normalize_ocr_output(ocr_result)
        quality = sum(len(line) for line in lines) + (avg_conf * 60.0)
        low_conf = avg_conf < 0.78
        if quality > best_quality:
            best_quality = quality
            best_lines = lines
            best_low_conf = low_conf

    return best_lines, best_low_conf


def normalize_image_for_ocr(image_bytes: bytes) -> bytes:
    if cv2 is None or np is None:
        return image_bytes
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        return image_bytes

    resized = resize_image_max_side(decoded, OCR_MAX_IMAGE_SIDE)
    ok, encoded = cv2.imencode(".png", resized)
    if not ok:
        return image_bytes
    return encoded.tobytes()


def resize_image_max_side(image: Any, max_side: int) -> Any:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_ocr_output(ocr_result: list) -> tuple[list[str], float]:
    lines: list[str] = []
    confidences: list[float] = []

    for row in ocr_result:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue

        rec_text = str(row[1]).strip()
        rec_score = 0.0
        if len(row) > 2:
            try:
                rec_score = float(row[2])
            except (TypeError, ValueError):
                rec_score = 0.0

        if rec_text:
            lines.append(rec_text)
            confidences.append(rec_score)

    if not confidences:
        return lines, 0.0
    return lines, sum(confidences) / len(confidences)


def detect_sections(page_texts: list[str], page_zone_texts: list[dict[str, str]] | None = None) -> list[str]:
    found: set[str] = set()
    for text in page_texts:
        for section_name, pattern in SECTION_PATTERNS.items():
            if pattern.search(text):
                found.add(section_name)
    if page_zone_texts:
        for zone_map in page_zone_texts:
            if zone_map.get("notes"):
                found.add("notes")
            if zone_map.get("revision_block"):
                found.add("revision_block")
            if zone_map.get("title_block"):
                found.add("title_block")
            if zone_map.get("drawing_area"):
                found.add("drawing_views")
    return sorted(found)


async def extract_context_rules(files: list[UploadFile]) -> list[ContextRule]:
    payloads = await read_uploaded_context_payloads(files)
    return extract_context_rules_from_payloads(payloads)


async def read_uploaded_context_payloads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    payloads: list[tuple[str, bytes]] = []
    for file in files:
        payloads.append((file.filename or "context.txt", await file.read()))
    return payloads


def extract_context_rules_from_payloads(payloads: list[tuple[str, bytes]]) -> list[ContextRule]:
    rules: list[ContextRule] = []
    for filename, data in payloads:
        lower_name = filename.lower()
        if lower_name.endswith(".csv"):
            rules.extend(parse_csv_rules(data, source_name=filename))
        elif lower_name.endswith(".xlsx"):
            rules.extend(parse_xlsx_rules(data, source_name=filename))
        elif lower_name.endswith(".json"):
            rules.extend(parse_json_rules(data, source_name=filename))
        elif lower_name.endswith(".docx"):
            rules.extend(parse_docx_rules(data, source_name=filename))
        elif lower_name.endswith(".txt"):
            rules.extend(parse_txt_rules(data, source_name=filename))
    return rules


def extract_context_entries_from_payloads(payloads: list[tuple[str, bytes]]) -> list[ContextEntry]:
    entries: list[ContextEntry] = []
    for filename, data in payloads:
        lower_name = filename.lower()
        try:
            if lower_name.endswith(".csv"):
                entries.extend(parse_csv_context_entries(data, source_name=filename))
            elif lower_name.endswith(".xlsx"):
                entries.extend(parse_xlsx_context_entries(data, source_name=filename))
            elif lower_name.endswith(".json"):
                entries.extend(parse_json_context_entries(data, source_name=filename))
            elif lower_name.endswith(".docx"):
                entries.extend(parse_docx_context_entries(data, source_name=filename))
            elif lower_name.endswith(".txt"):
                entries.extend(parse_txt_context_entries(data, source_name=filename))
        except Exception:
            continue
    return dedupe_context_entries(entries)


def extract_rules_from_foundational_docs(docs: list[FoundationalDoc]) -> list[ContextRule]:
    rules: list[ContextRule] = []
    for doc in docs:
        lowered = doc.source_name.lower()
        payload = doc.content.encode("utf-8", errors="ignore")
        try:
            if lowered.endswith(".csv"):
                rules.extend(parse_csv_rules(payload, source_name=doc.source_name))
            elif lowered.endswith(".xlsx"):
                rules.extend(parse_xlsx_rules(payload, source_name=doc.source_name))
            elif lowered.endswith(".json"):
                rules.extend(parse_json_rules(payload, source_name=doc.source_name))
            elif lowered.endswith(".docx"):
                rules.extend(parse_docx_rules(payload, source_name=doc.source_name))
            else:
                rules.extend(parse_txt_rules(payload, source_name=doc.source_name))
        except Exception:
            continue

    return rules


def extract_text_for_foundational_doc(raw: bytes, filename: str | None) -> str:
    lowered = (filename or "").lower()
    try:
        if lowered.endswith(".csv") or lowered.endswith(".txt") or lowered.endswith(".json"):
            return raw.decode("utf-8", errors="ignore")
        if lowered.endswith(".xlsx"):
            return extract_xlsx_text(raw)
        if lowered.endswith(".docx"):
            return extract_docx_text(raw)
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def ensure_surreal_configured() -> None:
    if not SURREAL_URL or not SURREAL_NS or not SURREAL_DB:
        raise HTTPException(
            status_code=503,
            detail="SurrealDB is not configured. Set SURREAL_URL, SURREAL_NS, and SURREAL_DB.",
        )
    if not surreal_auth_configured():
        raise HTTPException(
            status_code=503,
            detail="SurrealDB auth is not configured. Set SURREAL_TOKEN or SURREAL_USER/SURREAL_PASS.",
        )


def retrieve_foundational_docs_for_query(query: str, limit: int = 8) -> list[FoundationalDoc]:
    if not SURREAL_URL or not SURREAL_NS or not SURREAL_DB:
        return []
    if not surreal_auth_configured():
        return []
    try:
        chunks = fetch_recent_foundational_chunks(max_chunks=RAG_MAX_CHUNKS_SCAN)
        if chunks:
            ranked_chunks = rank_chunks_by_query(chunks, query)
            if ranked_chunks:
                selected = ranked_chunks[: max(4, limit * 3)]
                return aggregate_chunks_to_docs(selected, limit=max(1, limit))

        all_docs = fetch_recent_foundational_docs(max_docs=RAG_MAX_DOCS_SCAN)
        if not all_docs:
            return []

        scored = rank_docs_by_query(all_docs, query)
        return scored[: max(1, limit)]
    except Exception:
        # Analysis should continue even when Surreal retrieval is unavailable.
        return []


def rank_docs_by_query(docs: list[FoundationalDoc], query: str) -> list[FoundationalDoc]:
    terms = extract_terms(query, limit=24)
    if not terms:
        return docs[:8]

    scored: list[tuple[float, FoundationalDoc]] = []
    for doc in docs:
        text = doc.content.lower()
        overlap = 0
        for term in terms:
            if term in text:
                overlap += 1
        if overlap == 0:
            continue
        density = overlap / max(1, len(terms))
        scored.append((density, doc))

    scored.sort(key=lambda row: row[0], reverse=True)
    if not scored:
        return docs[:8]
    return [doc for _, doc in scored]


def rank_chunks_by_query(chunks: list[FoundationalChunk], query: str) -> list[FoundationalChunk]:
    terms = extract_terms(query, limit=30)
    if not terms:
        return chunks[:16]

    scored: list[tuple[float, FoundationalChunk]] = []
    for chunk in chunks:
        text = chunk.content.lower()
        chunk_terms = chunk.terms or []
        overlap = 0
        weighted = 0
        for term in terms:
            if term in text:
                overlap += 1
            if term in chunk_terms:
                weighted += 2

        if overlap == 0 and weighted == 0:
            continue

        coverage = overlap / max(1, len(terms))
        precision = overlap / max(1, len(set(chunk_terms)))
        score = coverage + (weighted / max(1, len(terms) * 2)) + (precision * 0.2)
        scored.append((score, chunk))

    scored.sort(key=lambda row: row[0], reverse=True)
    if not scored:
        return chunks[:16]
    return [chunk for _, chunk in scored]


def aggregate_chunks_to_docs(chunks: list[FoundationalChunk], limit: int) -> list[FoundationalDoc]:
    doc_map: dict[str, list[FoundationalChunk]] = {}
    for chunk in chunks:
        doc_map.setdefault(chunk.doc_id, []).append(chunk)

    docs: list[FoundationalDoc] = []
    for doc_id, group in doc_map.items():
        ordered = sorted(group, key=lambda c: c.chunk_index)
        source_name = ordered[0].source_name
        merged = "\n".join(chunk.content.strip() for chunk in ordered if chunk.content.strip())
        docs.append(FoundationalDoc(id=doc_id, source_name=source_name, content=truncate(merged, 9000)))

    docs.sort(key=lambda doc: len(doc.content), reverse=True)
    return docs[: max(1, limit)]


def chunk_text(content: str, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> list[str]:
    cleaned = re.sub(r"\n{3,}", "\n\n", content.strip())
    if not cleaned:
        return []

    if overlap >= chunk_size:
        overlap = max(50, chunk_size // 6)

    chunks: list[str] = []
    start = 0
    total = len(cleaned)
    while start < total:
        end = min(total, start + chunk_size)
        if end < total:
            next_break = cleaned.rfind("\n\n", start, end)
            if next_break > start + 200:
                end = next_break
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= total:
            break
        start = max(0, end - overlap)
    return chunks


def extract_terms(text: str, limit: int = 20) -> list[str]:
    candidates = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9\-_/.]{2,}\b", text.lower())
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "notes",
        "page",
        "drawing",
        "context",
        "item",
    }
    filtered = [token for token in candidates if token not in stop_words]
    counts = Counter(filtered)
    most_common = [token for token, _ in counts.most_common(limit)]
    return most_common


def parse_csv_rules(data: bytes, source_name: str | None = None) -> list[ContextRule]:
    df = pd.read_csv(io.BytesIO(data))
    normalized = {column.strip().lower(): column for column in df.columns}

    old_col = normalized.get("old") or normalized.get("deprecated") or normalized.get("current")
    new_col = normalized.get("new") or normalized.get("replacement") or normalized.get("updated")

    rules: list[ContextRule] = []
    if old_col and new_col:
        for _, row in df.iterrows():
            old_value = str(row.get(old_col, "")).strip()
            new_value = str(row.get(new_col, "")).strip()
            if old_value and new_value and old_value.lower() != "nan" and new_value.lower() != "nan":
                rules.append(ContextRule(old_value=old_value, new_value=new_value, source_name=source_name))

    return rules


def parse_csv_context_entries(data: bytes, source_name: str | None = None) -> list[ContextEntry]:
    df = pd.read_csv(io.BytesIO(data))
    normalized = {column.strip().lower(): column for column in df.columns}
    object_col = (
        normalized.get("object")
        or normalized.get("zone")
        or normalized.get("section")
        or normalized.get("category")
    )
    key_col = (
        normalized.get("key")
        or normalized.get("object_key")
        or normalized.get("field")
        or normalized.get("name")
    )
    value_col = (
        normalized.get("value")
        or normalized.get("expected")
        or normalized.get("target")
        or normalized.get("new")
        or normalized.get("replacement")
    )

    # Fallback to two-column key/value CSVs.
    if not key_col and not value_col and len(df.columns) >= 2:
        key_col = df.columns[0]
        value_col = df.columns[1]

    entries: list[ContextEntry] = []
    if not key_col or not value_col:
        return entries

    for idx, row in df.iterrows():
        key_raw = str(row.get(key_col, "")).strip()
        value_raw = str(row.get(value_col, "")).strip()
        object_raw = str(row.get(object_col, "")).strip() if object_col else ""
        if not key_raw or not value_raw or key_raw.lower() == "nan" or value_raw.lower() == "nan":
            continue
        entries.append(
            ContextEntry(
                object_name=normalize_key(object_raw) if object_raw else None,
                key=normalize_key(key_raw),
                value=value_raw,
                source_name=source_name,
                source_ref=f"row:{int(idx)+2}" if isinstance(idx, int) else None,
            )
        )
    return entries


def parse_json_rules(data: bytes, source_name: str | None = None) -> list[ContextRule]:
    payload = json.loads(data.decode("utf-8"))
    rules: list[ContextRule] = []

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            old_value = str(row.get("old") or row.get("deprecated") or "").strip()
            new_value = str(row.get("new") or row.get("replacement") or "").strip()
            if old_value and new_value:
                rules.append(ContextRule(old_value=old_value, new_value=new_value, source_name=source_name))

    return rules


def parse_json_context_entries(data: bytes, source_name: str | None = None) -> list[ContextEntry]:
    payload = json.loads(data.decode("utf-8"))
    entries: list[ContextEntry] = []

    def add_entry(object_name: str | None, key: str, value: Any) -> None:
        key_text = str(key).strip()
        value_text = stringify_context_value(value)
        if not key_text or not value_text:
            return
        entries.append(
            ContextEntry(
                object_name=normalize_key(object_name) if object_name else None,
                key=normalize_key(key_text),
                value=value_text,
                source_name=source_name,
            )
        )

    def walk(node: Any, object_name: str | None = None) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, object_name)
            return
        if not isinstance(node, dict):
            return

        # Direct schema support: {object, key, value}
        key_candidate = node.get("key") or node.get("object_key") or node.get("field")
        value_candidate = node.get("value") or node.get("expected") or node.get("target") or node.get("new")
        object_candidate = node.get("object") or node.get("zone") or node.get("section") or object_name
        if key_candidate is not None and value_candidate is not None:
            add_entry(str(object_candidate).strip() if object_candidate else None, str(key_candidate), value_candidate)

        # Generic nested object support: { notes: { material: "A36" } }
        for k, v in node.items():
            if k in {"object", "zone", "section", "key", "object_key", "field", "value", "expected", "target", "new"}:
                continue
            if isinstance(v, dict):
                walk(v, k)
            elif isinstance(v, list):
                if v and all(not isinstance(item, (dict, list)) for item in v):
                    add_entry(object_name, str(k), v)
                else:
                    walk(v, object_name)
            else:
                add_entry(object_name, str(k), v)

    walk(payload, None)
    return entries


def parse_txt_rules(data: bytes, source_name: str | None = None) -> list[ContextRule]:
    rules: list[ContextRule] = []
    text = data.decode("utf-8", errors="ignore")

    for line_idx, line in enumerate(text.splitlines(), start=1):
        parsed = parse_rule_line(line)
        if parsed:
            old_value, new_value = parsed
            rules.append(
                ContextRule(
                    old_value=old_value,
                    new_value=new_value,
                    source_name=source_name,
                    source_ref=f"line:{line_idx}",
                )
            )

    return rules


def parse_txt_context_entries(data: bytes, source_name: str | None = None) -> list[ContextEntry]:
    entries: list[ContextEntry] = []
    text = data.decode("utf-8", errors="ignore")
    for line_idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parsed = parse_generic_key_value_line(stripped)
        if not parsed:
            continue

        object_name, key, value = parsed
        if key and value:
            entries.append(
                ContextEntry(
                    object_name=normalize_key(object_name) if object_name else None,
                    key=normalize_key(key),
                    value=value,
                    source_name=source_name,
                    source_ref=f"line:{line_idx}",
                )
            )
    return entries


def parse_xlsx_rules(data: bytes, source_name: str | None = None) -> list[ContextRule]:
    rules: list[ContextRule] = []
    for sheet_name, row_idx, row in iterate_xlsx_rows(data):
        normalized = {str(k).strip().lower(): str(v).strip() for k, v in row.items() if str(k).strip()}
        old_value = (
            normalized.get("old")
            or normalized.get("deprecated")
            or normalized.get("current")
            or ""
        ).strip()
        new_value = (
            normalized.get("new")
            or normalized.get("replacement")
            or normalized.get("updated")
            or normalized.get("accepted")
            or ""
        ).strip()
        if old_value and new_value:
            rules.append(
                ContextRule(
                    old_value=old_value,
                    new_value=new_value,
                    source_name=source_name,
                    source_ref=f"{sheet_name}!row:{row_idx}",
                )
            )
    # Support simple paired-sheet workbooks like:
    # sheet OUTDATED: [value1, value2], sheet ACCEPTED: [value1, value2]
    if not rules:
        rules.extend(parse_xlsx_rules_from_sheet_pairs(data, source_name=source_name))
    return rules


def parse_xlsx_context_entries(data: bytes, source_name: str | None = None) -> list[ContextEntry]:
    entries: list[ContextEntry] = []
    for sheet_name, row_idx, row in iterate_xlsx_rows(data):
        normalized = {str(k).strip().lower(): str(v).strip() for k, v in row.items() if str(k).strip()}
        object_name = (
            normalized.get("object")
            or normalized.get("zone")
            or normalized.get("section")
            or normalized.get("category")
            or ""
        ).strip()
        key = (
            normalized.get("key")
            or normalized.get("object_key")
            or normalized.get("field")
            or normalized.get("name")
            or ""
        ).strip()
        value = (
            normalized.get("value")
            or normalized.get("expected")
            or normalized.get("target")
            or normalized.get("new")
            or normalized.get("replacement")
            or normalized.get("accepted")
            or ""
        ).strip()
        if not key or not value:
            non_empty = [(k, v) for k, v in normalized.items() if str(v).strip()]
            if len(non_empty) >= 2:
                key = key or str(non_empty[0][1]).strip()
                value = value or str(non_empty[1][1]).strip()
        if key and value:
            entries.append(
                ContextEntry(
                    object_name=normalize_key(object_name) if object_name else None,
                    key=normalize_key(key),
                    value=value,
                    source_name=source_name,
                    source_ref=f"{sheet_name}!row:{row_idx}",
                )
            )
    return entries


def parse_docx_rules(data: bytes, source_name: str | None = None) -> list[ContextRule]:
    text = extract_docx_text(data)
    return parse_txt_rules(text.encode("utf-8"), source_name=source_name)


def parse_docx_context_entries(data: bytes, source_name: str | None = None) -> list[ContextEntry]:
    text = extract_docx_text(data)
    return parse_txt_context_entries(text.encode("utf-8"), source_name=source_name)


def iterate_xlsx_rows(data: bytes) -> list[tuple[str, int, dict[str, Any]]]:
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str)
    except Exception:
        return []
    rows: list[tuple[str, int, dict[str, Any]]] = []
    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        for row_idx, row in df.fillna("").iterrows():
            rows.append((str(sheet_name), int(row_idx) + 2, row.to_dict()))
    return rows


def extract_xlsx_text(data: bytes) -> str:
    lines: list[str] = []
    for sheet_name, row_idx, row in iterate_xlsx_rows(data):
        parts = []
        for key, value in row.items():
            value_text = str(value).strip()
            if not value_text:
                continue
            parts.append(f"{str(key).strip()}: {value_text}")
        if parts:
            lines.append(f"[{sheet_name} row {row_idx}] " + " | ".join(parts))
    if not lines:
        sheet_values = extract_xlsx_sheet_values(data)
        for sheet_name, values in sheet_values.items():
            for idx, value in enumerate(values, start=1):
                lines.append(f"[{sheet_name} row {idx}] value: {value}")
    return "\n".join(lines)


def parse_xlsx_rules_from_sheet_pairs(data: bytes, source_name: str | None = None) -> list[ContextRule]:
    sheet_values = extract_xlsx_sheet_values(data)
    if not sheet_values:
        return []

    outdated_name = first_matching_sheet_name(sheet_values.keys(), {"outdated", "deprecated", "obsolete", "old"})
    accepted_name = first_matching_sheet_name(sheet_values.keys(), {"accepted", "approved", "new", "current"})

    if not outdated_name or not accepted_name:
        names = list(sheet_values.keys())
        if len(names) >= 2:
            outdated_name, accepted_name = names[0], names[1]
        else:
            return []

    outdated_values = [v for v in sheet_values.get(outdated_name, []) if v.strip()]
    accepted_values = [v for v in sheet_values.get(accepted_name, []) if v.strip()]
    pair_count = min(len(outdated_values), len(accepted_values))
    rules: list[ContextRule] = []
    for idx in range(pair_count):
        old_value = outdated_values[idx].strip()
        new_value = accepted_values[idx].strip()
        if not old_value or not new_value:
            continue
        rules.append(
            ContextRule(
                old_value=old_value,
                new_value=new_value,
                source_name=source_name,
                source_ref=f"{outdated_name}/{accepted_name}!row:{idx+1}",
            )
        )
    return rules


def first_matching_sheet_name(sheet_names: Any, targets: set[str]) -> str | None:
    for name in sheet_names:
        normalized = normalize_key(str(name))
        if normalized in targets:
            return str(name)
    return None


def extract_xlsx_sheet_values(data: bytes) -> dict[str, list[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            workbook_xml = archive.read("xl/workbook.xml").decode("utf-8", errors="ignore")
            rels_xml = archive.read("xl/_rels/workbook.xml.rels").decode("utf-8", errors="ignore")
            shared_strings = extract_shared_strings(archive)
            sheet_targets = parse_workbook_sheet_targets(workbook_xml, rels_xml)
            result: dict[str, list[str]] = {}
            for sheet_name, target in sheet_targets:
                path = f"xl/{target}".replace("xl/xl/", "xl/")
                if path not in archive.namelist():
                    continue
                sheet_xml = archive.read(path).decode("utf-8", errors="ignore")
                values = extract_sheet_cell_values(sheet_xml, shared_strings)
                if values:
                    result[sheet_name] = values
            return result
    except Exception:
        return {}


def extract_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    xml_data = archive.read("xl/sharedStrings.xml").decode("utf-8", errors="ignore")
    try:
        root = ET.fromstring(xml_data)
    except Exception:
        return []
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: list[str] = []
    for si in root.findall(".//a:si", ns):
        parts = [node.text or "" for node in si.findall(".//a:t", ns)]
        strings.append("".join(parts).strip())
    return strings


def parse_workbook_sheet_targets(workbook_xml: str, rels_xml: str) -> list[tuple[str, str]]:
    ns_main = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ns_rel = {"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    relmap: dict[str, str] = {}
    try:
        rels_root = ET.fromstring(rels_xml)
        for rel in rels_root.findall(".//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"):
            rid = rel.attrib.get("Id")
            target = rel.attrib.get("Target")
            if rid and target:
                relmap[rid] = target
    except Exception:
        return []
    output: list[tuple[str, str]] = []
    try:
        wb_root = ET.fromstring(workbook_xml)
        for sheet in wb_root.findall(".//a:sheets/a:sheet", ns_main):
            name = sheet.attrib.get("name", "").strip()
            rid = sheet.attrib.get(f"{{{ns_rel['r']}}}id")
            if name and rid and rid in relmap:
                output.append((name, relmap[rid]))
    except Exception:
        return []
    return output


def extract_sheet_cell_values(sheet_xml: str, shared_strings: list[str]) -> list[str]:
    try:
        root = ET.fromstring(sheet_xml)
    except Exception:
        return []
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values: list[str] = []
    for cell in root.findall(".//a:sheetData/a:row/a:c", ns):
        cell_type = cell.attrib.get("t", "")
        v_node = cell.find("a:v", ns)
        if v_node is None or v_node.text is None:
            continue
        raw = v_node.text.strip()
        if not raw:
            continue
        value = raw
        if cell_type == "s":
            try:
                value = shared_strings[int(raw)]
            except Exception:
                value = raw
        value = value.strip()
        if value:
            values.append(value)
    return values


def extract_docx_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml_data = archive.read("word/document.xml")
    except Exception:
        return ""
    try:
        root = ET.fromstring(xml_data)
    except Exception:
        return ""

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    lines: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        chunks = []
        for text_node in paragraph.findall(".//w:t", ns):
            if text_node.text:
                chunks.append(text_node.text)
        merged = "".join(chunks).strip()
        if merged:
            lines.append(merged)
    return "\n".join(lines)


def parse_rule_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if "=>" in stripped:
        left, right = [chunk.strip() for chunk in stripped.split("=>", maxsplit=1)]
        if left and right:
            return left, right
    if "->" in stripped:
        left, right = [chunk.strip() for chunk in stripped.split("->", maxsplit=1)]
        if left and right:
            return left, right
    return None


def parse_generic_key_value_line(line: str) -> tuple[str | None, str, str] | None:
    # Supports:
    # - zone.key: value
    # - key: value
    # - zone.key => value
    # - key => value
    match = re.match(r"^(?:(?P<object>[A-Za-z0-9 _\-/]+)\.)?(?P<key>[A-Za-z0-9 _\-/]+)\s*(?::|=>)\s*(?P<value>.+)$", line)
    if not match:
        return None
    object_name = match.group("object")
    key = (match.group("key") or "").strip()
    value = (match.group("value") or "").strip()
    if not key or not value:
        return None
    return object_name.strip() if object_name else None, key, value


def stringify_context_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " | ".join(parts)
    if isinstance(value, (dict,)):
        try:
            return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            return str(value).strip()
    return str(value).strip()


def dedupe_context_entries(entries: list[ContextEntry]) -> list[ContextEntry]:
    deduped: list[ContextEntry] = []
    seen: set[tuple[str | None, str, str, str | None]] = set()
    for entry in entries:
        key = (entry.object_name, entry.key, entry.value, entry.source_name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def compare_zone_rows_to_context(
    zone_rows: list[ZoneItemRow],
    context_entries: list[ContextEntry],
    rules: list[ContextRule],
) -> list[ComparisonRow]:
    by_object_and_key: dict[tuple[str, str], list[ContextEntry]] = {}
    by_key: dict[str, list[ContextEntry]] = {}
    for entry in context_entries:
        by_key.setdefault(entry.key, []).append(entry)
        if entry.object_name:
            by_object_and_key.setdefault((entry.object_name, entry.key), []).append(entry)

    comparisons: list[ComparisonRow] = []
    for row in zone_rows:
        drawing_value = " | ".join(row.object_values).strip()
        zone_key = normalize_key(row.zone)
        candidates = by_object_and_key.get((zone_key, row.object_key), []) or by_key.get(row.object_key, [])
        selected = candidates[0] if candidates else None

        status: Literal["change", "review_only", "unsure"] = "unsure"
        replace_with = "NO CHANGE RECOMMENDED"
        change_type = "Review Only"
        source_basis = "No governing source found"
        priority = "—"
        notes = "Not found in context files."

        if selected:
            if values_equivalent(row.object_values, selected.value):
                status = "review_only"
                replace_with = "NO CHANGE"
                source_basis = format_source_basis(selected.source_name, selected.source_ref, "Accepted in context")
                priority = "Low"
                notes = "Matches accepted value from context."
            else:
                status = "change"
                replace_with = selected.value
                change_type = "Value Update"
                source_basis = format_source_basis(selected.source_name, selected.source_ref, "Context value differs")
                priority = "High"
                notes = "Context indicates this field is outdated or superseded."
        else:
            matched_rule = find_matching_rule(row.object_values, rules)
            if matched_rule:
                status = "change"
                replace_with = matched_rule.new_value
                change_type = "Spec/Standard Update"
                source_basis = format_source_basis(
                    matched_rule.source_name,
                    matched_rule.source_ref,
                    "Rule-based outdated to approved replacement",
                )
                priority = "High"
                notes = "Detected outdated content that has an approved replacement."

        comparisons.append(
            ComparisonRow(
                sheet=row.page,
                zone=row.zone,
                existing_text=drawing_value,
                replace_with=replace_with,
                change_type=change_type,
                source_basis=source_basis,
                priority=priority,
                notes=notes,
                status=status,
            )
        )

    return comparisons


def filter_rows_for_display(rows: list[ComparisonRow]) -> list[ComparisonRow]:
    return [row for row in rows if row.status != "unsure"]


def find_matching_rule(found_values: list[str], rules: list[ContextRule]) -> ContextRule | None:
    for value in found_values:
        normalized_value = normalize_compare_text(value)
        for rule in rules:
            if normalize_compare_text(rule.old_value) in normalized_value:
                return rule
    return None


def format_source_basis(source_name: str | None, source_ref: str | None, default_reason: str) -> str:
    if source_name and source_ref:
        return f"{source_name} ({source_ref}) - {default_reason}"
    if source_name:
        return f"{source_name} - {default_reason}"
    return default_reason


def values_equivalent(found_values: list[str], context_value: str) -> bool:
    norm_context = normalize_compare_text(context_value)
    if not norm_context:
        return False
    for value in found_values:
        if normalize_compare_text(value) == norm_context:
            return True
    return False


def normalize_compare_text(value: str) -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def suggest_value_from_rules(found_values: list[str], rules: list[ContextRule]) -> str | None:
    matched = find_matching_rule(found_values, rules)
    if matched:
        return matched.new_value
    return None


def find_outdated_references(page_texts: list[str], rules: list[ContextRule]) -> list[ComplianceIssue]:
    issues: list[ComplianceIssue] = []

    for page_idx, text in enumerate(page_texts, start=1):
        for match in STANDARD_PATTERN.findall(text):
            for rule in rules:
                if rule.old_value.lower() in match.lower():
                    issues.append(
                        ComplianceIssue(
                            id=f"std-{page_idx}-{len(issues)+1}",
                            issue_type="outdated_standard",
                            severity="high",
                            message="Detected outdated standard reference in drawing.",
                            page=page_idx,
                            section="notes",
                            evidence=match,
                            expected_value=rule.new_value,
                            found_value=rule.old_value,
                            recommendation="Update the standard reference to the approved replacement.",
                        )
                    )

        for rule in rules:
            if rule.old_value.lower() in text.lower() and rule.new_value.lower() not in text.lower():
                issues.append(
                    ComplianceIssue(
                        id=f"spec-{page_idx}-{len(issues)+1}",
                        issue_type="outdated_spec",
                        severity="medium",
                        message="Detected spec text that appears outdated relative to uploaded context.",
                        page=page_idx,
                        section="notes",
                        evidence=truncate(rule.old_value),
                        expected_value=rule.new_value,
                        found_value=rule.old_value,
                        recommendation="Replace outdated wording and re-run validation.",
                    )
                )

    return dedupe_issues(issues)


def find_bom_view_mismatches(page_texts: list[str]) -> list[ComplianceIssue]:
    issues: list[ComplianceIssue] = []
    bom_items: set[str] = set()
    callout_items: set[str] = set()

    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            # Demo-friendly BOM row parsing: "12 P/N1234 BRACKET 2"
            match = re.match(r"^(\d{1,3})\s+\S+\s+.*\s+(\d+)$", line)
            if match:
                bom_items.add(match.group(1))

        for item in ITEM_CALLOUT_PATTERN.findall(text):
            callout_items.add(item)

    for item in sorted(callout_items):
        if bom_items and item not in bom_items:
            issues.append(
                ComplianceIssue(
                    id=f"bom-{item}",
                    issue_type="bom_mismatch",
                    severity="high",
                    message="Drawing view callout item is missing from BOM.",
                    page=None,
                    section="drawing_views",
                    evidence=f"ITEM {item}",
                    expected_value="Item exists in BOM",
                    found_value="Missing from BOM",
                    recommendation="Add missing BOM row or remove incorrect callout in drawing view.",
                )
            )

    return issues


def dedupe_issues(issues: list[ComplianceIssue]) -> list[ComplianceIssue]:
    seen: set[tuple[str, int | None, str]] = set()
    deduped: list[ComplianceIssue] = []

    for issue in issues:
        key = (issue.issue_type, issue.page, issue.evidence)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)

    return deduped


def truncate(text: str, max_len: int = 90) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def find_llm_change_rows(
    page_texts: list[str],
    page_zone_texts: list[dict[str, str]],
    context_rules: list[ContextRule],
    context_entries: list[ContextEntry],
    context_payloads: list[tuple[str, bytes]],
    sections: list[str],
    llm_runtime: LlmRuntimeConfig,
    context_sources: list[str],
) -> tuple[list[ComparisonRow], bool, str | None]:
    if not OLLAMA_ENABLED:
        return [], False, None
    if llm_runtime.mode == "online" and not llm_runtime.api_key:
        return [], False, "Online mode requires an Ollama API key."

    prompt = build_llm_change_prompt(
        page_texts=page_texts,
        page_zone_texts=page_zone_texts,
        rules=context_rules,
        entries=context_entries,
        sections=sections,
        context_payloads=context_payloads,
        context_sources=context_sources,
    )
    response_text, error_message = call_ollama(prompt, llm_runtime)
    if error_message:
        return [], False, error_message

    payload = parse_json_object(response_text)
    if not payload:
        return [], False, "Gemma response could not be parsed as JSON."
    raw_rows = payload.get("rows", [])
    if not isinstance(raw_rows, list):
        return [], False, "Gemma JSON did not include a valid 'rows' array."

    rows: list[ComparisonRow] = []
    for raw in raw_rows:
        row = normalize_change_row(raw)
        if row:
            rows.append(row)
    if not rows:
        return [], True, "Gemma returned no valid change rows."
    return rows, True, None


def normalize_change_row(raw: Any) -> ComparisonRow | None:
    if not isinstance(raw, dict):
        return None
    try:
        sheet_raw = raw.get("sheet")
        if sheet_raw is None:
            sheet_raw = raw.get("page")
        sheet = int(sheet_raw)
        zone = str(raw.get("zone") or "").strip() or "drawing_area"
        existing_text = str(raw.get("existing_text") or raw.get("existing_text_field") or "").strip()
        if not existing_text:
            return None
        replace_with = str(raw.get("replace_with") or "NO CHANGE RECOMMENDED").strip() or "NO CHANGE RECOMMENDED"
        change_type = str(raw.get("change_type") or "Review Only").strip() or "Review Only"
        source_basis = str(raw.get("source_basis") or "No governing source found").strip() or "No governing source found"
        priority = str(raw.get("priority") or "—").strip() or "—"
        notes = str(raw.get("notes") or "").strip()
        status_text = str(raw.get("status") or "").strip().lower()
        status: Literal["change", "review_only", "unsure"] = "unsure"
        if status_text in {"change", "update", "revise"}:
            status = "change"
        elif status_text in {"review_only", "review", "no_change"}:
            status = "review_only"
        elif "no governing source" in source_basis.lower():
            status = "unsure"
        elif replace_with.upper() == "NO CHANGE":
            status = "review_only"
        elif replace_with.upper() == "NO CHANGE RECOMMENDED":
            status = "unsure"
        else:
            status = "change"

        return ComparisonRow(
            sheet=sheet,
            zone=zone,
            existing_text=existing_text,
            replace_with=replace_with,
            change_type=change_type,
            source_basis=source_basis,
            priority=priority,
            notes=notes,
            status=status,
        )
    except Exception:
        return None


def build_llm_change_prompt(
    page_texts: list[str],
    page_zone_texts: list[dict[str, str]],
    rules: list[ContextRule],
    entries: list[ContextEntry],
    sections: list[str],
    context_payloads: list[tuple[str, bytes]],
    context_sources: list[str],
) -> str:
    sampled_pages = []
    for idx, text in enumerate(page_texts[:4], start=1):
        sampled_pages.append(f"SHEET {idx}:\n{truncate(text, 3400)}")

    zone_pages = []
    for idx, zone_map in enumerate(page_zone_texts[:4], start=1):
        zone_lines = [f"SHEET {idx} ZONES:"]
        for zone_name in ["notes", "revision_block", "title_block", "drawing_area"]:
            zone_text = zone_map.get(zone_name, "").strip()
            if zone_text:
                zone_lines.append(f"- {zone_name}: {truncate(zone_text, 1200)}")
        if len(zone_lines) > 1:
            zone_pages.append("\n".join(zone_lines))

    rules_text = "\n".join(
        [
            f"- {rule.old_value} => {rule.new_value} (source: {rule.source_name or 'unknown'})"
            for rule in rules[:80]
        ]
    ) or "- none"
    entries_text = "\n".join(
        [
            f"- object={entry.object_name or 'n/a'} key={entry.key} value={truncate(entry.value, 200)} "
            f"(source: {entry.source_name or 'unknown'})"
            for entry in entries[:120]
        ]
    ) or "- none"
    context_text = build_context_payload_excerpt(context_payloads)
    sources_text = ", ".join(context_sources) if context_sources else "none"
    sections_text = ", ".join(sections) if sections else "none"
    drawing_excerpt = "\n\n".join(sampled_pages)
    zone_excerpt = "\n\n".join(zone_pages) if zone_pages else "none"

    return f"""
You are an engineering drawing comparison assistant.

Core instruction:
Parse the content in this drawing by section (notes, revision history block, title block, drawing area) and compare against the provided context files (excel/doc/csv/json/txt) that may indicate values as OUTDATED or ACCEPTED.
If something is not found in context files, treat it as unsure and do not suggest a change.
If context indicates an update, suggest the needed change.

Return STRICT JSON only with this shape:
{{
  "rows": [
    {{
      "sheet": 1,
      "zone": "notes|revision_block|title_block|drawing_area",
      "existing_text": "string",
      "replace_with": "string (NO CHANGE or NO CHANGE RECOMMENDED allowed)",
      "change_type": "string",
      "source_basis": "must include context filename(s) when possible",
      "priority": "High|Medium|Low|—",
      "notes": "string",
      "status": "change|review_only|unsure"
    }}
  ]
}}

Requirements:
- Prioritize evidence-backed updates.
- Include Source/Basis using filename list whenever possible.
- For unsure rows, keep status=unsure and replace_with=NO CHANGE RECOMMENDED.
- Keep rows concise and table-ready.

Detected sections: {sections_text}
Context source files: {sources_text}

Context rules:
{rules_text}

Context entries:
{entries_text}

Context text excerpts:
{context_text}

Detected layout zones:
{zone_excerpt}

Drawing text:
{drawing_excerpt}
""".strip()


def build_context_payload_excerpt(payloads: list[tuple[str, bytes]]) -> str:
    excerpts: list[str] = []
    for filename, data in payloads[:8]:
        lowered = filename.lower()
        try:
            if lowered.endswith(".xlsx"):
                text = extract_xlsx_text(data)
            elif lowered.endswith(".docx"):
                text = extract_docx_text(data)
            else:
                text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        if not text.strip():
            continue
        excerpts.append(f"[{filename}]\n{truncate(text, 1200)}")
    return "\n\n".join(excerpts) if excerpts else "- none"


def find_llm_issues(
    page_texts: list[str],
    page_zone_texts: list[dict[str, str]],
    rules: list[ContextRule],
    sections: list[str],
    llm_runtime: LlmRuntimeConfig,
) -> tuple[list[ComplianceIssue], bool, str | None]:
    if not OLLAMA_ENABLED:
        return [], False, None
    if llm_runtime.mode == "online" and not llm_runtime.api_key:
        return [], False, "Online mode requires an Ollama API key."

    prompt = build_llm_prompt(page_texts, page_zone_texts, rules, sections)
    response_text, error_message = call_ollama(prompt, llm_runtime)
    if error_message:
        return [], False, error_message

    payload = parse_json_object(response_text)
    if not payload:
        return [], False, "Gemma response could not be parsed as JSON."

    raw_issues = payload.get("issues", [])
    if not isinstance(raw_issues, list):
        return [], False, "Gemma JSON did not include a valid 'issues' array."

    issues: list[ComplianceIssue] = []
    for idx, raw in enumerate(raw_issues, start=1):
        if not isinstance(raw, dict):
            continue

        raw.setdefault("id", f"llm-{idx}")
        raw.setdefault("recommendation", "Review and update drawing content per approved context.")
        raw.setdefault("section", "notes")
        raw.setdefault("severity", "medium")
        raw.setdefault("issue_type", "missing_reference")
        raw.setdefault("message", "LLM-detected compliance concern.")
        raw.setdefault("page", None)
        raw.setdefault("evidence", "Generated from drawing/context comparison.")
        raw.setdefault("expected_value", None)
        raw.setdefault("found_value", None)

        try:
            issues.append(ComplianceIssue(**raw))
        except Exception:
            continue

    return dedupe_issues(issues), True, None


def build_llm_prompt(
    page_texts: list[str],
    page_zone_texts: list[dict[str, str]],
    rules: list[ContextRule],
    sections: list[str],
) -> str:
    sampled_pages = []
    for idx, text in enumerate(page_texts[:4], start=1):
        sampled_pages.append(f"PAGE {idx}:\n{truncate(text, 3500)}")

    zone_pages = []
    for idx, zone_map in enumerate(page_zone_texts[:4], start=1):
        zone_lines = [f"PAGE {idx} ZONES:"]
        for zone_name in ["notes", "revision_block", "title_block", "drawing_area"]:
            zone_text = zone_map.get(zone_name, "").strip()
            if zone_text:
                zone_lines.append(f"- {zone_name}: {truncate(zone_text, 1200)}")
        if len(zone_lines) > 1:
            zone_pages.append("\n".join(zone_lines))

    rules_text = "\n".join([f"- {rule.old_value} => {rule.new_value}" for rule in rules[:50]])
    sections_text = ", ".join(sections) if sections else "none"
    drawing_excerpt = "\n\n".join(sampled_pages)
    zone_excerpt = "\n\n".join(zone_pages) if zone_pages else "none"

    return f"""
You are a drawing compliance checker. Compare drawing content to approved context.
Return STRICT JSON only.

Required JSON shape:
{{
  "issues": [
    {{
      "id": "string",
      "issue_type": "outdated_standard|outdated_spec|material_mismatch|bom_mismatch|missing_reference|ocr_uncertain",
      "severity": "low|medium|high|critical",
      "message": "string",
      "page": 1,
      "section": "notes|revision_block|title_block|bom|drawing_views",
      "evidence": "string",
      "expected_value": "string|null",
      "found_value": "string|null",
      "recommendation": "string"
    }}
  ]
}}

Only report real, evidence-backed issues.
If there are no issues, return: {{"issues":[]}}

Detected sections: {sections_text}
Context rules:
{rules_text or "- none"}

Detected layout zones:
{zone_excerpt}

Drawing text:
{drawing_excerpt}
""".strip()


def resolve_llm_runtime(inference_mode: Literal["online", "local"], request_key: str | None) -> LlmRuntimeConfig:
    cleaned_key = request_key.strip() if request_key else None
    if inference_mode == "online":
        return LlmRuntimeConfig(
            mode="online",
            model=OLLAMA_CLOUD_MODEL,
            url=OLLAMA_CLOUD_URL,
            api_key=cleaned_key or OLLAMA_API_KEY,
        )

    return LlmRuntimeConfig(
        mode="local",
        model=OLLAMA_LOCAL_MODEL,
        url=OLLAMA_LOCAL_URL,
        api_key=None,
    )


def call_ollama(prompt: str, llm_runtime: LlmRuntimeConfig) -> tuple[str, str | None]:
    payload = {
        "model": llm_runtime.model,
        "messages": [
            {
                "role": "system",
                "content": "You output only compact JSON and no extra text.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    try:
        headers = {"Content-Type": "application/json"}
        if llm_runtime.api_key:
            headers["Authorization"] = f"Bearer {llm_runtime.api_key}"

        req = request.Request(
            llm_runtime.url,
            method="POST",
            headers=headers,
            data=json.dumps(payload).encode("utf-8"),
        )
        with request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        content = parsed.get("message", {}).get("content", "")
        if not content:
            return "", "Ollama returned an empty response."
        return content, None
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return "", f"Ollama HTTP error {exc.code}: {detail}"
    except error.URLError:
        if llm_runtime.mode == "local":
            return "", "Could not connect to local Ollama on localhost:11434."
        return "", "Could not connect to Ollama Cloud API."
    except TimeoutError:
        return "", f"Ollama timed out after {OLLAMA_TIMEOUT_SECONDS:.0f}s."
    except Exception as exc:
        return "", f"Ollama call failed: {exc}"


def store_foundational_doc(source_name: str, content: str, terms: list[str]) -> None:
    escaped_source = sql_quote(source_name)
    escaped_content = sql_quote(content)
    terms_sql = ", ".join(sql_quote(term) for term in terms)
    query = (
        f"CREATE {SURREAL_TABLE} SET "
        f"source_name = {escaped_source}, "
        f"content = {escaped_content}, "
        f"terms = [{terms_sql}], "
        f"created_at = time::now();"
    )
    result = surreal_query(query)
    doc_id = extract_created_id(result)
    if not doc_id:
        return
    store_foundational_chunks(doc_id=doc_id, source_name=source_name, content=content)


def extract_created_id(payload: list[dict[str, Any]]) -> str | None:
    if not payload:
        return None
    first_result = payload[0].get("result")
    if isinstance(first_result, list) and first_result:
        first = first_result[0]
        if isinstance(first, dict) and first.get("id"):
            return str(first["id"])
    if isinstance(first_result, dict) and first_result.get("id"):
        return str(first_result["id"])
    return None


def store_foundational_chunks(doc_id: str, source_name: str, content: str) -> None:
    chunks = chunk_text(content)
    if not chunks:
        return

    escaped_doc_id = sql_quote(doc_id)
    escaped_source_name = sql_quote(source_name)
    for idx, chunk in enumerate(chunks):
        chunk_terms = extract_terms(chunk, limit=24)
        terms_sql = ", ".join(sql_quote(term) for term in chunk_terms)
        chunk_query = (
            f"CREATE {SURREAL_CHUNK_TABLE} SET "
            f"doc_id = {escaped_doc_id}, "
            f"source_name = {escaped_source_name}, "
            f"chunk_index = {idx}, "
            f"content = {sql_quote(chunk)}, "
            f"terms = [{terms_sql}], "
            f"created_at = time::now();"
        )
        surreal_query(chunk_query)


def store_zone_rows(source_name: str, rows: list[ZoneItemRow]) -> None:
    escaped_source = sql_quote(source_name)
    for row in rows:
        values_sql = ", ".join(sql_quote(value) for value in row.object_values)
        table_name = zone_table_name(row.zone)
        query = (
            f"CREATE {table_name} SET "
            f"source_name = {escaped_source}, "
            f"page = {row.page}, "
            f"zone = {sql_quote(row.zone)}, "
            f"object_key = {sql_quote(row.object_key)}, "
            f"object_values = [{values_sql}], "
            f"line_number = {row.line_number}, "
            f"created_at = time::now();"
        )
        surreal_query(query)


def zone_table_name(zone: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", zone.lower()).strip("_") or "unknown"
    return f"{SURREAL_ZONE_TABLE}_{normalized}"


def fetch_recent_foundational_docs(max_docs: int = 250) -> list[FoundationalDoc]:
    query = (
        f"SELECT id, source_name, content, created_at "
        f"FROM {SURREAL_TABLE} "
        f"ORDER BY created_at DESC "
        f"LIMIT {max(1, min(max_docs, 500))};"
    )
    payload = surreal_query(query)
    rows = payload[0].get("result", []) if payload else []
    docs: list[FoundationalDoc] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        docs.append(
            FoundationalDoc(
                id=str(row.get("id", "")),
                source_name=str(row.get("source_name", "unknown")),
                content=str(row.get("content", "")),
            )
        )
    return docs


def fetch_recent_foundational_chunks(max_chunks: int = 1200) -> list[FoundationalChunk]:
    query = (
        f"SELECT id, doc_id, source_name, chunk_index, content, terms, created_at "
        f"FROM {SURREAL_CHUNK_TABLE} "
        f"ORDER BY created_at DESC "
        f"LIMIT {max(1, min(max_chunks, 5000))};"
    )
    payload = surreal_query(query)
    rows = payload[0].get("result", []) if payload else []
    chunks: list[FoundationalChunk] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_terms = row.get("terms")
        terms: list[str] = []
        if isinstance(raw_terms, list):
            terms = [str(term) for term in raw_terms if str(term).strip()]

        chunks.append(
            FoundationalChunk(
                id=str(row.get("id", "")),
                doc_id=str(row.get("doc_id", "")),
                source_name=str(row.get("source_name", "unknown")),
                chunk_index=int(row.get("chunk_index", 0)),
                content=str(row.get("content", "")),
                terms=terms,
            )
        )
    return chunks


def surreal_query(sql: str) -> list[dict[str, Any]]:
    base = SURREAL_URL.rstrip("/")
    if not base:
        raise HTTPException(status_code=503, detail="SurrealDB URL is not configured.")

    if base.endswith("/sql"):
        url = base
    elif base.endswith("/rpc"):
        url = base[:-4] + "/sql"
    else:
        url = base + "/sql"

    headers = {
        "Content-Type": "text/plain",
        "Accept": "application/json",
        "NS": SURREAL_NS,
        "DB": SURREAL_DB,
    }

    if SURREAL_TOKEN:
        headers["Authorization"] = f"Bearer {SURREAL_TOKEN}"
    elif SURREAL_USER and SURREAL_PASS:
        credentials = f"{SURREAL_USER}:{SURREAL_PASS}".encode("utf-8")
        token = base64_encode(credentials)
        headers["Authorization"] = f"Basic {token}"

    req = request.Request(
        url,
        method="POST",
        headers=headers,
        data=sql.encode("utf-8"),
    )
    try:
        with request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if exc.code in (401, 403):
            raise HTTPException(
                status_code=503,
                detail="SurrealDB authentication/permissions failed. Check SURREAL_TOKEN or SURREAL_USER/SURREAL_PASS permissions.",
            ) from exc
        compact = compact_http_error_detail(detail)
        if compact:
            raise HTTPException(status_code=502, detail=f"SurrealDB HTTP error {exc.code}: {compact}") from exc
        raise HTTPException(status_code=502, detail=f"SurrealDB HTTP error {exc.code}.") from exc
    except error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach SurrealDB: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="SurrealDB request timed out.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SurrealDB query failed: {exc}") from exc

    if not isinstance(parsed, list):
        raise HTTPException(status_code=500, detail="Unexpected SurrealDB response format.")
    return [row for row in parsed if isinstance(row, dict)]


def base64_encode(raw: bytes) -> str:
    import base64

    return base64.b64encode(raw).decode("ascii")


def sql_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def parse_json_object(text: str) -> dict | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def compact_http_error_detail(raw: str, limit: int = 220) -> str:
    if not raw:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return truncate(cleaned, max_len=limit) if cleaned else ""


def sanitize_service_error(detail: Any, service: str) -> str:
    text = compact_http_error_detail(str(detail), limit=180)
    if not text:
        return f"{service} is unavailable. Continued without foundational context."
    return f"{service} is unavailable ({text}). Continued without foundational context."


def surreal_auth_configured() -> bool:
    return bool(SURREAL_TOKEN or (SURREAL_USER and SURREAL_PASS))
