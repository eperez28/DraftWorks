from __future__ import annotations

import io
import json
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal
from urllib import error, request

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


class AnalysisResult(BaseModel):
    run_id: str
    summary: str
    sections_detected: list[str]
    issues: list[ComplianceIssue]
    meta: AnalysisMeta


@dataclass
class ContextRule:
    old_value: str
    new_value: str


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
    page_texts, ocr_low_conf_pages = extract_drawing_text(
        drawing_bytes=drawing_bytes,
        filename=drawing.filename,
        content_type=drawing.content_type,
    )
    sections = detect_sections(page_texts)

    context_rules = await extract_context_rules(context_files)
    if use_foundational_context:
        query_text = " ".join(page_texts[:2])[:6000]
        foundational_docs = retrieve_foundational_docs_for_query(query_text, limit=8)
        context_rules.extend(extract_rules_from_foundational_docs(foundational_docs))
    issues: list[ComplianceIssue] = []
    llm_used = False
    llm_error: str | None = None

    issues.extend(find_outdated_references(page_texts, context_rules))
    issues.extend(find_bom_view_mismatches(page_texts))
    llm_runtime = resolve_llm_runtime(inference_mode, ollama_api_key)
    llm_issues, llm_used, llm_error = find_llm_issues(page_texts, context_rules, sections, llm_runtime)
    issues.extend(llm_issues)

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
        ),
    )


def extract_drawing_text(
    drawing_bytes: bytes,
    filename: str | None,
    content_type: str | None,
) -> tuple[list[str], list[int]]:
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


def extract_pdf_text_with_ocr(pdf_bytes: bytes) -> tuple[list[str], list[int]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_texts: list[str] = []
    low_conf_pages: list[int] = []

    for i, page in enumerate(doc):
        text = page.get_text("text") or ""

        if len(text.strip()) >= 100:
            page_texts.append(text)
            continue

        if ocr_engine is None:
            page_texts.append(text)
            low_conf_pages.append(i + 1)
            continue

        pix = page.get_pixmap(dpi=300)
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

    return page_texts, low_conf_pages


def extract_image_text_with_ocr(image_bytes: bytes) -> tuple[list[str], list[int]]:
    if ocr_engine is None:
        return [""], [1]

    lines, low_conf = run_ocr_with_preprocessing(image_bytes)
    extracted = "\n".join(lines).strip()
    if not extracted:
        return [""], [1]
    if low_conf:
        return [extracted], [1]
    return [extracted], []


def run_ocr_with_preprocessing(image_bytes: bytes) -> tuple[list[str], bool]:
    candidates = [image_bytes]

    if cv2 is not None and np is not None:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is not None:
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


def detect_sections(page_texts: list[str]) -> list[str]:
    found: set[str] = set()
    for text in page_texts:
        for section_name, pattern in SECTION_PATTERNS.items():
            if pattern.search(text):
                found.add(section_name)
    return sorted(found)


async def extract_context_rules(files: list[UploadFile]) -> list[ContextRule]:
    rules: list[ContextRule] = []

    for file in files:
        data = await file.read()
        lower_name = file.filename.lower() if file.filename else ""

        if lower_name.endswith(".csv"):
            rules.extend(parse_csv_rules(data))
        elif lower_name.endswith(".json"):
            rules.extend(parse_json_rules(data))
        elif lower_name.endswith(".txt"):
            rules.extend(parse_txt_rules(data))

    return rules


def extract_rules_from_foundational_docs(docs: list[FoundationalDoc]) -> list[ContextRule]:
    rules: list[ContextRule] = []
    for doc in docs:
        lowered = doc.source_name.lower()
        payload = doc.content.encode("utf-8", errors="ignore")
        try:
            if lowered.endswith(".csv"):
                rules.extend(parse_csv_rules(payload))
            elif lowered.endswith(".json"):
                rules.extend(parse_json_rules(payload))
            else:
                rules.extend(parse_txt_rules(payload))
        except Exception:
            continue

    return rules


def extract_text_for_foundational_doc(raw: bytes, filename: str | None) -> str:
    lowered = (filename or "").lower()
    try:
        if lowered.endswith(".csv") or lowered.endswith(".txt") or lowered.endswith(".json"):
            return raw.decode("utf-8", errors="ignore")
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def ensure_surreal_configured() -> None:
    if not SURREAL_URL or not SURREAL_NS or not SURREAL_DB:
        raise HTTPException(
            status_code=503,
            detail="SurrealDB is not configured. Set SURREAL_URL, SURREAL_NS, and SURREAL_DB.",
        )


def retrieve_foundational_docs_for_query(query: str, limit: int = 8) -> list[FoundationalDoc]:
    if not SURREAL_URL or not SURREAL_NS or not SURREAL_DB:
        return []

    all_docs = fetch_recent_foundational_docs(max_docs=250)
    if not all_docs:
        return []

    scored = rank_docs_by_query(all_docs, query)
    return scored[: max(1, limit)]


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


def parse_csv_rules(data: bytes) -> list[ContextRule]:
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
                rules.append(ContextRule(old_value=old_value, new_value=new_value))

    return rules


def parse_json_rules(data: bytes) -> list[ContextRule]:
    payload = json.loads(data.decode("utf-8"))
    rules: list[ContextRule] = []

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            old_value = str(row.get("old") or row.get("deprecated") or "").strip()
            new_value = str(row.get("new") or row.get("replacement") or "").strip()
            if old_value and new_value:
                rules.append(ContextRule(old_value=old_value, new_value=new_value))

    return rules


def parse_txt_rules(data: bytes) -> list[ContextRule]:
    rules: list[ContextRule] = []
    text = data.decode("utf-8", errors="ignore")

    for line in text.splitlines():
        if "=>" in line:
            old_value, new_value = [chunk.strip() for chunk in line.split("=>", maxsplit=1)]
            if old_value and new_value:
                rules.append(ContextRule(old_value=old_value, new_value=new_value))

    return rules


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


def find_llm_issues(
    page_texts: list[str],
    rules: list[ContextRule],
    sections: list[str],
    llm_runtime: LlmRuntimeConfig,
) -> tuple[list[ComplianceIssue], bool, str | None]:
    if not OLLAMA_ENABLED:
        return [], False, None
    if llm_runtime.mode == "online" and not llm_runtime.api_key:
        return [], False, "Online mode requires an Ollama API key."

    prompt = build_llm_prompt(page_texts, rules, sections)
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


def build_llm_prompt(page_texts: list[str], rules: list[ContextRule], sections: list[str]) -> str:
    sampled_pages = []
    for idx, text in enumerate(page_texts[:4], start=1):
        sampled_pages.append(f"PAGE {idx}:\n{truncate(text, 3500)}")

    rules_text = "\n".join([f"- {rule.old_value} => {rule.new_value}" for rule in rules[:50]])
    sections_text = ", ".join(sections) if sections else "none"
    drawing_excerpt = "\n\n".join(sampled_pages)

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
    surreal_query(query)


def fetch_recent_foundational_docs(max_docs: int = 250) -> list[FoundationalDoc]:
    query = (
        f"SELECT id, source_name, content "
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
        raise HTTPException(status_code=502, detail=f"SurrealDB HTTP error {exc.code}: {detail}") from exc
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
