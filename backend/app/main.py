from __future__ import annotations

import io
import json
import re
import uuid
from dataclasses import dataclass
from typing import Literal

import fitz
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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


SECTION_PATTERNS = {
    "notes": re.compile(r"\bnotes?\b", re.IGNORECASE),
    "title_block": re.compile(r"\btitle\s*block\b|\bdrawing\s*title\b", re.IGNORECASE),
    "revision_block": re.compile(r"\brev(?:ision)?\b|\bchange\s*log\b", re.IGNORECASE),
    "bom": re.compile(r"\bbill\s+of\s+materials\b|\bbom\b|\bparts\s+list\b", re.IGNORECASE),
    "drawing_views": re.compile(r"\bsection\s+[a-z]\b|\bview\b|\bdetail\b", re.IGNORECASE),
}

STANDARD_PATTERN = re.compile(r"\b(?:ASTM|MIL-STD|ISO|SAE)[\s\-]*[A-Z0-9\-.]+\b")
ITEM_CALLOUT_PATTERN = re.compile(r"\b(?:ITEM|ITM|BALLOON)\s*#?\s*(\d{1,3})\b", re.IGNORECASE)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(
    drawing: UploadFile = File(...),
    context_files: list[UploadFile] = File(default=[]),
    use_foundational_context: bool = Form(default=False),
) -> AnalysisResult:
    pdf_bytes = await drawing.read()
    page_texts, ocr_low_conf_pages = extract_pdf_text_with_ocr(pdf_bytes)
    sections = detect_sections(page_texts)

    context_rules = await extract_context_rules(context_files)
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
        meta=AnalysisMeta(
            pages_processed=len(page_texts),
            used_foundational_context=use_foundational_context,
            context_files_count=len(context_files),
        ),
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

        pix = page.get_pixmap(dpi=250)
        ocr_result, _ = ocr_engine(pix.tobytes("png"))

        if not ocr_result:
            page_texts.append(text)
            low_conf_pages.append(i + 1)
            continue

        lines: list[str] = []
        low_conf = False
        for row in ocr_result:
            rec_text = row[1]
            rec_score = row[2]
            lines.append(rec_text)
            if rec_score < 0.78:
                low_conf = True

        page_texts.append("\n".join(lines))
        if low_conf:
            low_conf_pages.append(i + 1)

    return page_texts, low_conf_pages


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
