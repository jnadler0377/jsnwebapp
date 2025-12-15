from __future__ import annotations

import io
import json
import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from app.models import Case, Note
from app.database import engine
from app.services.skiptrace_service import (
    load_property_for_case,
    load_skiptrace_for_case,
)

logger = logging.getLogger("pascowebapp")

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_ROOT = BASE_DIR / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------- Formatting helpers ---------------- #

def _fmt_money(raw: Any) -> str:
    if raw is None or raw == "":
        return "-"
    try:
        if isinstance(raw, (int, float)):
            return f"${float(raw):,.2f}"
        s = str(raw).strip()
        if not s:
            return "-"
        cleaned = s.replace("$", "").replace(",", "")
        val = float(cleaned)
        return f"${val:,.2f}"
    except Exception:
        return str(raw)


def _parse_float(raw: Any, default: float = 0.0) -> float:
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        s = str(raw).strip()
        if not s:
            return default
        cleaned = s.replace("$", "").replace(",", "")
        return float(cleaned)
    except Exception:
        return default


def _yn_icon(val: Any) -> str:
    # ✓ / ✗ like in your sample
    if isinstance(val, bool):
        return "✓" if val else "✗"
    if isinstance(val, (int, float)):
        return "✓" if float(val) != 0 else "✗"
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"y", "yes", "true", "1"}:
            return "✓"
        if v in {"n", "no", "false", "0"}:
            return "✗"
    return "✗"


def _fmt_phone(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 10:
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:]}"
    if len(digits) == 11 and digits[0] == "1":
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    return s


def _resolve_address(case: Case, primary_prop: Optional[dict]) -> str:
    # 1) Manual override
    override = (getattr(case, "address_override", "") or "").strip()
    if override:
        return override.upper()

    # 2) BatchData address
    addr = (primary_prop or {}).get("address") or {}
    if addr:
        street = (
            addr.get("street")
            or addr.get("streetNoUnit")
            or addr.get("line1")
            or ""
        ).strip()
        city = (addr.get("city") or "").strip()
        state = (addr.get("state") or "").strip()
        zip_code = (addr.get("zip") or "").strip()
        parts = [p for p in [street, city, state] if p]
        full = ", ".join(parts)
        if zip_code:
            full = f"{full}, {zip_code}".strip()
        if full:
            return full.upper()

    # 3) Case.address
    addr_attr = (getattr(case, "address", "") or "").strip()
    if addr_attr:
        return addr_attr.upper()

    # 4) Fallback to parcel
    parcel = (getattr(case, "parcel_id", "") or "").strip()
    if parcel:
        return f"(Parcel {parcel})"

    return "(ADDRESS NOT SET)"


def _sum_liens_for_calc(case: Case) -> float:
    raw = getattr(case, "outstanding_liens", "[]") or "[]"
    try:
        liens = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        liens = []

    total = 0.0
    if isinstance(liens, list):
        for item in liens:
            if not isinstance(item, dict):
                continue
            amt = (
                item.get("amount")
                or item.get("balance")
                or item.get("lien_amount")
            )
            total += _parse_float(amt, 0.0)
    return round(total, 2)


def _iter_liens_for_display(case: Case) -> List[Dict[str, Any]]:
    raw = getattr(case, "outstanding_liens", "[]") or "[]"
    try:
        liens = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        liens = []

    rows: List[Dict[str, Any]] = []
    if isinstance(liens, list):
        for item in liens:
            if not isinstance(item, dict):
                continue
            desc_raw = (
                item.get("description")
                or item.get("lien_type")
                or item.get("type")
                or item.get("creditor")
                or ""
            )
            desc = str(desc_raw).strip() or "Lien"
            amount = item.get("amount") or item.get("balance") or item.get("lien_amount")
            rows.append({"description": desc, "amount": amount})
    return rows


def _append_pdf_if_exists(writer: PdfWriter, rel_path: Optional[str]):
    if not rel_path:
        return
    try:
        full = UPLOAD_ROOT / rel_path
        if not full.exists():
            return
        reader = PdfReader(str(full))
        for page in reader.pages:
            writer.add_page(page)
    except Exception as exc:
        logger.warning("Failed to append PDF %s: %s", rel_path, exc)


# ---------------- Text layout helpers (wrapping) ---------------- #

def _wrap_text(c: canvas.Canvas, text: str, max_width: float, font_name: str, font_size: int) -> List[str]:
    if not text:
        return [""]
    c.setFont(font_name, font_size)
    words = text.split()
    lines: List[str] = []
    current = ""
    for w in words:
        test = (current + " " + w).strip()
        if not current:
            current = w
            continue
        if c.stringWidth(test, font_name, font_size) <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


class SimpleLayout:
    """
    Small helper to keep line spacing and page breaks sane, while
    matching your single-page style.
    """

    def __init__(self, c: canvas.Canvas, page_size=letter, margin_x: int = 50, margin_bottom: int = 50):
        self.c = c
        self.width, self.height = page_size
        self.margin_x = margin_x
        self.margin_bottom = margin_bottom
        self.y = self.height - 50

    def _ensure_space(self, needed: float = 18):
        if self.y - needed < self.margin_bottom:
            self.c.showPage()
            self.y = self.height - 50

    def line(self, text: str = "", size: int = 10, bold: bool = False, leading: int = 14):
        max_width = self.width - self.margin_x * 2
        font = "Helvetica-Bold" if bold else "Helvetica"
        wrapped = _wrap_text(self.c, text or "", max_width, font, size)
        for ln in wrapped:
            self._ensure_space(leading + 2)
            self.c.setFont(font, size)
            self.c.drawString(self.margin_x, self.y, ln)
            self.y -= leading

    def section_title(self, title: str):
        # Blank line then bold section name (no underline – to match sample)
        self.line("", size=6)
        self.line(title, size=11, bold=True)


# ---------------- Skip-trace loading & normalization ---------------- #

def _load_skiptrace_for_report(case_id: int) -> Optional[Any]:
    """
    Load skip-trace data for the report.

    Priority:
      1) Normalized skip-trace tables via load_skiptrace_for_case
      2) Legacy cases.skip_trace_json column (if present)
    """
    # First: try the normalized tables
    try:
        data = load_skiptrace_for_case(case_id)
        if data:
            return data
    except Exception as exc:
        logger.warning("load_skiptrace_for_case failed for case %s: %s", case_id, exc)

    # Fallback: legacy JSON on the cases table
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT skip_trace_json FROM cases WHERE id = :id"),
                {"id": case_id},
            ).mappings().first()
        if row and row.get("skip_trace_json"):
            try:
                return json.loads(row["skip_trace_json"])
            except Exception as exc:
                logger.warning(
                    "Failed to parse skip_trace_json for case %s in report: %s",
                    case_id,
                    exc,
                )
    except Exception as exc:
        logger.warning(
            "Failed report skip-trace fallback for case %s: %s",
            case_id,
            exc,
        )

    return None


def _extract_skiptrace_summary(data: Any) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Try hard to extract:
      - owner_name
      - list of phone dicts
      - list of email dicts

    Handles several shapes:
      - {'contacts': [ {...} ]}
      - {'owners': [ {...} ]}
      - {'results': [ {'persons' or 'people' or 'owners': [ {...} ]} ]}
      - {'primary_owner': {...}}
      - {'phones': [...], 'emails': [...]}
      - [ {...}, {...} ]  (list of contacts)
    """
    owner_name = ""
    phones: List[Dict[str, Any]] = []
    emails: List[Dict[str, Any]] = []

    def add_phone_entry(p: Any):
        if isinstance(p, dict):
            if p.get("number"):
                phones.append(p)
        elif p:
            phones.append({"number": p})

    def add_email_entry(e: Any):
        if isinstance(e, dict):
            if e.get("email"):
                emails.append(e)
        elif e:
            emails.append({"email": e})

    def process_contact(contact: Dict[str, Any]):
        nonlocal owner_name
        if not isinstance(contact, dict):
            return
        name = (
            contact.get("full_name")
            or contact.get("name")
            or contact.get("ownerName")
            or ""
        ).strip()
        if name and not owner_name:
            owner_name = name

        for p in contact.get("phones") or []:
            add_phone_entry(p)
        for e in contact.get("emails") or []:
            add_email_entry(e)

    if not data:
        return owner_name, phones, emails

    # If it's already a list, treat as contacts list
    if isinstance(data, list):
        for c in data:
            if isinstance(c, dict):
                process_contact(c)
        return owner_name, phones, emails

    if not isinstance(data, dict):
        return owner_name, phones, emails

    # contacts
    if isinstance(data.get("contacts"), list):
        for c in data["contacts"]:
            process_contact(c)

    # owners
    if isinstance(data.get("owners"), list):
        for c in data["owners"]:
            process_contact(c)

    # primary_owner
    if isinstance(data.get("primary_owner"), dict):
        process_contact(data["primary_owner"])

    # results -> persons/people/owners
    if isinstance(data.get("results"), list):
        for r in data["results"]:
            if not isinstance(r, dict):
                continue
            persons = (
                r.get("persons")
                or r.get("people")
                or r.get("owners")
                or []
            )
            for c in persons:
                process_contact(c)

    # top-level phones/emails
    for p in data.get("phones") or []:
        add_phone_entry(p)
    for e in data.get("emails") or []:
        add_email_entry(e)

    return owner_name, phones, emails


# ---------------- Main entrypoint ---------------- #

def generate_case_report(case_id: int, db: Session) -> StreamingResponse:
    """
    Generate a JSN Holdings Case Report matching the sample layout:
      - Header
      - Deal Summary
      - Outstanding Liens
      - Defendants
      - Skip Trace (Owner & Contact Details)
      - Mortgage Snapshot
      - Notes
      - Attached Documents
    """

    # Fetch case
    getter = getattr(db, "get", None)
    if callable(getter):
        case = db.get(Case, case_id)
    else:
        case = db.query(Case).get(case_id)  # type: ignore[call-arg]
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Property / BatchData (for address if needed)
    primary_prop = None
    try:
        payload = load_property_for_case(case_id)
        results = (payload or {}).get("results") or {}
        props = results.get("properties") or []
        if props:
            primary_prop = props[0]
    except Exception as exc:
        logger.warning("Failed to load property for case %s: %s", case_id, exc)

    # Basic fields
    case_number = getattr(case, "case_number", "") or ""
    style = getattr(case, "style", "") or ""
    filing_dt = getattr(case, "filing_datetime", "") or ""
    parcel_id = getattr(case, "parcel_id", "") or ""
    address_str = _resolve_address(case, primary_prop)

    # Deal numbers
    arv_num = _parse_float(getattr(case, "arv", None), 0.0)
    rehab_num = _parse_float(getattr(case, "rehab", None), 0.0)

    closing_raw = getattr(case, "closing_costs", None)
    try:
        user_closing = float(closing_raw) if closing_raw not in (None, "") else None
    except Exception:
        user_closing = None
    closing_num = user_closing if user_closing is not None else round(arv_num * 0.045, 2)

    # ✅ Correct JSN Max Offer: (ARV * 70%) - Rehab - Closing
    jsn_max_offer = round(
        max(0.0, (arv_num * 0.70) - rehab_num - closing_num),
        2,
    )

    total_liens = _sum_liens_for_calc(case)
    seller_cash = max(0.0, jsn_max_offer - total_liens)

    lien_rows = _iter_liens_for_display(case)

    # Defendants
    defendants: List[str] = []
    rel_defendants = getattr(case, "defendants", None)
    if rel_defendants:
        try:
            for d in rel_defendants:
                name = (getattr(d, "name", "") or "").strip()
                if name:
                    defendants.append(name)
        except Exception:
            pass

    # Notes
    try:
        notes: List[Note] = (
            db.query(Note)
            .filter(Note.case_id == case_id)
            .order_by(Note.id.desc())
            .all()
        )
    except Exception:
        notes = []

    # Skip-trace: use robust loader + normalizer
    skip_raw = _load_skiptrace_for_report(case_id)
    primary_owner_name, phones_block, emails_block = _extract_skiptrace_summary(skip_raw)

    # ---------- Build the summary PDF matching your sample ---------- #
    summary_buf = io.BytesIO()
    c = canvas.Canvas(summary_buf, pagesize=letter)
    layout = SimpleLayout(c)

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(layout.margin_x, layout.y, "JSN Holdings – Case Report")
    layout.y -= 18
    c.setFont("Helvetica", 10)
    generated_str = _dt.datetime.now().strftime("%m/%d/%Y")
    c.drawString(layout.margin_x, layout.y, f"Generated: {generated_str}")
    layout.y -= 18

    # Case line, address line, etc.
    layout.line(f"Case {case_number}", size=11, bold=False)
    layout.line(f"Address: {address_str}", size=11)
    pf_line = f"Parcel ID: {parcel_id}"
    if filing_dt:
        pf_line += f" | Filing Date: {filing_dt}"
    layout.line(pf_line, size=10)
    layout.line(f"Style / Case Name: {style}", size=10)

    # Deal Summary section
    layout.section_title("Deal Summary")
    layout.line(f"ARV: {_fmt_money(arv_num)}")
    layout.line(f"Rehab: {_fmt_money(rehab_num)}")
    layout.line(f"Closing Costs: {_fmt_money(closing_num)}")
    layout.line(f"JSN Max Offer: {_fmt_money(jsn_max_offer)}")
    layout.line(f"Max Seller in Hand Cash (after liens): {_fmt_money(seller_cash)}")

    # Outstanding Liens section
    layout.section_title("Outstanding Liens")
    if lien_rows:
        for row in lien_rows:
            desc = row["description"]
            amt = _fmt_money(row["amount"])
            layout.line(f"• {desc} – {amt}")
    else:
        layout.line("No outstanding liens recorded.")

    # Defendants section
    layout.section_title("Defendants")
    if defendants:
        for d in defendants:
            layout.line(f"• {d}")
    else:
        layout.line("No defendants recorded.")

    # Skip Trace (Owner & Contact Details)
    layout.section_title("Skip Trace (Owner & Contact Details)")
    if primary_owner_name:
        layout.line(f"Primary Owner: {primary_owner_name}")
    else:
        layout.line("Primary Owner: (not found)")

    # Phones
    if phones_block:
        layout.line("Phones:")
        for ph in phones_block:
            num = _fmt_phone(ph.get("number"))
            if not num:
                continue
            pieces = [num]
            if ph.get("type"):
                pieces.append(f"({ph['type']})")
            if ph.get("score") is not None:
                pieces.append(f"Score: {ph['score']}")
            if ph.get("last_reported"):
                pieces.append(f"Last: {ph['last_reported']}")
            pieces.append(f"Tested: {_yn_icon(ph.get('tested'))}")
            pieces.append(f"Reachable: {_yn_icon(ph.get('reachable'))}")
            pieces.append(f"DNC: {_yn_icon(ph.get('dnc'))}")
            layout.line(" - " + " | ".join(pieces))
    else:
        layout.line("Phones: (none found)")

    # Emails
    if emails_block:
        layout.line("Emails:")
        for em in emails_block:
            addr = (em.get("email") or "").strip()
            if not addr:
                continue
            pieces = [addr]
            tested = em.get("deliverable")
            if tested is not None:
                pieces.append(f"Tested: {_yn_icon(tested)}")
            layout.line(" - " + " | ".join(pieces))
    else:
        layout.line("Emails: (none found)")

        # Notes
    layout.section_title("Notes")
    if notes:
        for n in notes[:15]:
            created = getattr(n, "created_at", "") or ""
            content = (getattr(n, "content", "") or "").replace("\r", " ").replace("\n", " ")
            if len(content) > 220:
                content = content[:217] + "..."
            layout.line(f"• [{created}] {content}", size=9)
    else:
        layout.line("No notes recorded.")

    # Attached Documents
    layout.section_title("Attached Documents")
    attached_labels: List[str] = []
    vc = getattr(case, "verified_complaint_path", "") or ""
    mtg = getattr(case, "mortgage_path", "") or ""
    cd = getattr(case, "current_deed_path", "") or ""
    pd = getattr(case, "previous_deed_path", "") or ""
    val = getattr(case, "value_calc_path", "") or ""

    if vc:
        attached_labels.append("Verified Complaint")
    if mtg:
        attached_labels.append("Mortgage")
    if cd:
        attached_labels.append("Current Deed")
    if pd:
        attached_labels.append("Previous Deed")
    if val:
        attached_labels.append("Value Calculation")

    if attached_labels:
        for lbl in attached_labels:
            layout.line(f"• {lbl}")
    else:
        layout.line("No documents uploaded.")

    # Finish summary page(s)
    c.showPage()
    c.save()
    summary_buf.seek(0)

    # ---------- Combine with uploaded PDFs ---------- #
    writer = PdfWriter()
    try:
        sr = PdfReader(summary_buf)
        for p in sr.pages:
            writer.add_page(p)
    except Exception as exc:
        logger.warning("Failed to read summary PDF for case %s: %s", case_id, exc)
        out = io.BytesIO(summary_buf.getvalue())
        out.seek(0)
        return StreamingResponse(
            out,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename=case_{case_id}_report.pdf"},
        )

    _append_pdf_if_exists(writer, vc)
    _append_pdf_if_exists(writer, mtg)
    _append_pdf_if_exists(writer, cd)
    _append_pdf_if_exists(writer, pd)
    _append_pdf_if_exists(writer, val)

    out_buf = io.BytesIO()
    if len(writer.pages) == 0:
        out_buf.write(summary_buf.getvalue())
        out_buf.seek(0)
    else:
        writer.write(out_buf)
        out_buf.seek(0)

    return StreamingResponse(
        out_buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=case_{case_id}_report.pdf"},
    )
