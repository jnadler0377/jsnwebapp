from __future__ import annotations

# ---- Windows event loop fix for asyncio subprocess ----
import sys as _sys
import asyncio as _asyncio
if _sys.platform == "win32":
    try:
        _asyncio.set_event_loop_policy(_asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# ---------------- Stdlib ----------------
import logging
import asyncio
import csv as _csv
import datetime as _dt
import os
import sys
import tempfile
import uuid
import io, json
from pathlib import Path
from typing import List, Optional
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from tools.import_pasco_csv import main as import_pasco_csv_main

# ---------------- FastAPI / Responses ----------------
from fastapi import (
    FastAPI,
    Request,
    Depends,
    UploadFile,
    File,
    Form,
    Query,
    HTTPException,
)
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------- DB / ORM ----------------
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text, bindparam  # add text, bindparam
from sqlalchemy.exc import OperationalError

# ---------------- App imports ----------------
from app.services.progress_bus import progress_bus
from app.settings import settings
from .database import Base, engine, SessionLocal
from .models import Case, Defendant, Docket, Note
from .utils import ensure_case_folder, compute_offer_70
from .schemas import OutstandingLien, OutstandingLiensUpdate  # NEW

# ======================================================================
# App bootstrap
# ======================================================================
app = FastAPI(title="JSN Holdings Foreclosure Manager")
logger = logging.getLogger("pascowebapp")
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
UPLOAD_ROOT = BASE_DIR / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_ROOT)), name="uploads")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ======================================================================
# Jinja filters / globals
# ======================================================================
def _currency(v):
    try:
        return "${:,.2f}".format(float(v))
    except Exception:
        return "$0.00"


def streetview_url(address: str) -> str:
    """
    Prefer Google Street View Static API if key present, else fall back
    to Static Map with a marker. Reads key from settings (env/.env).
    """
    if not address:
        return ""
    key = settings.GOOGLE_MAPS_API_KEY
    if key:
        base = "https://maps.googleapis.com/maps/api/streetview"
        return f"{base}?size=600x360&location={address}&key={key}"
    base = "https://maps.googleapis.com/maps/api/staticmap"
    return f"{base}?size=640x360&markers={address}"


def _parcel_to_property_card_param(parcel_id: str | None) -> Optional[str]:
    """
    Convert Pasco parcel formats to the property card 'parcel=' digits string.

    Example:
      Input:  '33-24-16-0260-00000-2540'
      Output: '1624330260000002540'
      (the first three 2-digit sets are mirrored: 33-24-16 -> 16 24 33)
    """
    if not parcel_id:
        return None

    s = parcel_id.strip().replace(" ", "")
    parts = s.split("-")

    # If it looks like the standard dash-delimited format with first three 2-digit parts
    if len(parts) >= 3 and all(len(p) == 2 for p in parts[:3]):
        reordered = parts[2] + parts[1] + parts[0] + "".join(parts[3:])
        digits = "".join(ch for ch in reordered if ch.isdigit())
        return digits or None

    # Fallback: digits only
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits or None


def pasco_appraiser_url(parcel_id: str | None) -> Optional[str]:
    """Return the direct property card URL for a given parcel id."""
    param = _parcel_to_property_card_param(parcel_id)
    if not param:
        return None
    return f"https://search.pascopa.com/parcel.aspx?parcel={param}"


templates.env.filters["currency"] = _currency
templates.env.globals["streetview_url"] = streetview_url
templates.env.globals["pasco_appraiser_url"] = pasco_appraiser_url

# ======================================================================
# DB session
# ======================================================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


Base.metadata.create_all(bind=engine)

# ======================================================================
# Startup: ensure late-added columns exist (sqlite ALTERs)
# ======================================================================
@app.on_event("startup")
def ensure_sqlite_columns():
    Base.metadata.create_all(bind=engine)
    try:
        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("cases")}
        with engine.begin() as conn:
            if "current_deed_path" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE cases ADD COLUMN current_deed_path TEXT DEFAULT ''"
                )
            if "previous_deed_path" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE cases ADD COLUMN previous_deed_path TEXT DEFAULT ''"
                )
            # NEW: Outstanding liens column (JSON stored as TEXT)
            if "outstanding_liens" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE cases ADD COLUMN outstanding_liens TEXT DEFAULT '[]'"
                )
    except OperationalError:
        # first run or non-sqlite; ignore
        pass

# ======================================================================
# Helpers: shell runner + scraper glue
# ======================================================================
async def run_command_with_logs(cmd: list[str], job_id: str) -> int:
    """
    Async process runner that streams stdout->progress_bus line by line.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    async for raw in proc.stdout:
        await progress_bus.publish(job_id, raw.decode(errors="ignore").rstrip("\\n"))
    rc = await proc.wait()
    await progress_bus.publish(job_id, f"[done] exit_code={rc}")
    return rc


def _find_scraper_script() -> Path:
    """
    Locate `pasco_foreclosure_scraper.py` in either:
    - <root>/Pasco Foreclosure Scrape
    - <root>/app/scrapers
    """
    candidates = [
        BASE_DIR / "Pasco Foreclosure Scrape" / "pasco_foreclosure_scraper.py",
        BASE_DIR / "app" / "scrapers" / "pasco_foreclosure_scraper.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise HTTPException(
        status_code=500,
        detail=(
            "Scraper script not found. Place 'pasco_foreclosure_scraper.py' in "
            "'Pasco Foreclosure Scrape/' or 'app/scrapers/'."
        ),
    )


def _import_csv_into_db(db: Session, csv_path: str) -> tuple[int, int, int]:
    """
    Lightweight importer (upsert by case_number, ignore duplicates).
    Returns (added, updated, skipped).
    """
    import re
    logger = logging.getLogger(__name__)

    def norm_case(s):
        s = str(s or "").strip().replace("\\", "-").replace("/", "-")
        s = re.sub(r"\s+", "", s)
        return s

    def pick_col(headers, candidates):
        """
        Given a list of headers and candidate names, return the first
        header that matches (case-insensitive, trimmed).
        """
        norm_headers = [h.strip().lower() for h in headers]
        for cand in candidates:
            cand_norm = cand.strip().lower()
            if cand_norm in norm_headers:
                # return the original header name exactly as in the CSV
                return headers[norm_headers.index(cand_norm)]
        return None

    added, updated, skipped = 0, 0, 0

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f)
        headers = reader.fieldnames or []
        logger.info("UpdateCases: CSV headers = %s", headers)

        # Try multiple possible names for important columns
        case_col   = pick_col(headers, ["Case #", "Case Number", "Case", "Case No.", "Case No"])
        filing_col = pick_col(headers, ["Filing Date", "Filing", "Filed"])
        style_col  = pick_col(headers, ["Case Name", "Style", "Case Style"])

        if not case_col:
            msg = f"Could not find case number column in CSV headers: {headers}"
            logger.error("UpdateCases: %s", msg)
            # fail cleanly so you see the error on the progress page
            raise HTTPException(status_code=400, detail=msg)

        for row in reader:
            cn_raw = row.get(case_col, "")
            cn = norm_case(cn_raw)
            if not cn:
                skipped += 1
                continue

            case = db.query(Case).filter(Case.case_number == cn).one_or_none()
            if not case:
                case = Case(case_number=cn)
                if filing_col:
                    case.filing_datetime = row.get(filing_col, "") or None
                if style_col:
                    case.style = row.get(style_col, "") or None
                db.add(case)
                db.flush()
                added += 1
            else:
                # only fill blanks to avoid overwriting your manual edits
                if not case.filing_datetime and filing_col:
                    case.filing_datetime = row.get(filing_col, "") or None
                if not case.style and style_col:
                    case.style = row.get(style_col, "") or None
                updated += 1

            # defendants: add only new names
            dnames = [
                row.get(k, "")
                for k in row.keys()
                if k and k.strip().lower().startswith("defendant")
            ]
            dnames = [d.strip() for d in dnames if d and d.strip()]

            existing = {d.name for d in (case.defendants or [])}
            for name in dnames:
                if name not in existing:
                    db.add(Defendant(case_id=case.id, name=name))

        db.commit()

    logger.info(
        "UpdateCases: Import complete. Added=%s Updated=%s Skipped=%s",
        added, updated, skipped,
    )
    return added, updated, skipped


# ======================================================================
# Routes: home, list, detail
# ======================================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse(url="/cases", status_code=303)


# @app.get("/cases", response_class=HTMLResponse)
#def cases_list(request: Request, page: int = Query(1), db: Session = Depends(get_db)):
#    qry = db.query(Case)
#    page_size = 10
#    total = qry.count()
#    pages = (total + page_size - 1) // page_size
#    logger.info("Cases pagination: total=%s pages=%s page=%s", total, pages, page)
#    offset = (page - 1) * page_size
#    cases = qry.order_by(Case.filing_datetime.desc()).offset(offset).limit(page_size).all()
#    pagination = {"page": page, "pages": pages, "total": total}
#    return templates.TemplateResponse(
#        "cases_list.html",
#        {"request": request, "cases": cases, "pagination": pagination},
#    )


# Case detail — explicitly fetch notes and attach to `case` so template can use `case.notes`
@app.get("/cases/new", response_class=HTMLResponse)
def new_case_form(request: Request):
    return templates.TemplateResponse("cases_new.html", {"request": request, "error": None})
@app.post("/cases/create")
def create_case(
    request: Request,
    case_number: str = Form(...),
    filing_date: Optional[str] = Form(None),   # "YYYY-MM-DD" or blank
    style: Optional[str] = Form(None),
    parcel_id: Optional[str] = Form(None),
    address_override: Optional[str] = Form(None),
    arv: Optional[str] = Form(None),            # <-- accept as str
    rehab: Optional[str] = Form(None),          # <-- accept as str
    closing_costs: Optional[str] = Form(None),  # <-- accept as str
    defendants_csv: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    # helpers
    def _num(x: Optional[str]) -> Optional[float]:
        if x is None:
            return None
        s = x.strip()
        if not s:
            return None
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return None

    cn = (case_number or "").strip()
    if not cn:
        return templates.TemplateResponse("cases_new.html", {"request": request, "error": "Case # is required."})

    # Duplicate check
    exists = db.query(Case).filter(Case.case_number == cn).one_or_none()
    if exists:
        return templates.TemplateResponse("cases_new.html", {"request": request, "error": f"Case {cn} already exists (ID {exists.id})."})

    # Create case
    case = Case(case_number=cn)
    if filing_date:
        case.filing_datetime = filing_date.strip()
    if style: case.style = style.strip()
    if parcel_id: case.parcel_id = parcel_id.strip()
    if address_override: case.address_override = address_override.strip()

    # only set if provided
    v_arv = _num(arv)
    v_rehab = _num(rehab)
    v_cc = _num(closing_costs)
    if v_arv is not None: case.arv = v_arv
    if v_rehab is not None: case.rehab = v_rehab
    if v_cc is not None: case.closing_costs = v_cc

    db.add(case)
    db.flush()

    if defendants_csv:
        raw = defendants_csv.replace("\r", "\n")
        parts = [p.strip() for chunk in raw.split("\n") for p in chunk.split(",")]
        seen = set()
        for name in parts:
            if name and name not in seen:
                seen.add(name)
                db.add(Defendant(case_id=case.id, name=name))

    db.commit()
    return RedirectResponse(url=f"/cases/{case.id}", status_code=303)


@app.get("/cases/{case_id}", response_class=HTMLResponse)
def case_detail(request: Request, case_id: int, db: Session = Depends(get_db)):
    getter = getattr(db, "get", None)
    if callable(getter):
        case = db.get(Case, case_id)
    else:
        case = db.query(Case).get(case_id)  # type: ignore[call-arg]

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Clean up defendants list (hide blank / nan / nil)
    try:
        raw_defs = list(getattr(case, "defendants", []) or [])
        cleaned_defs = []
        for d in raw_defs:
            name = (getattr(d, "name", "") or "").strip()
            if not name:
                continue
            lower = name.lower()
            if lower in {"nan", "none", "nil"}:
                continue
            cleaned_defs.append(d)
        try:
            setattr(case, "defendants", cleaned_defs)
        except Exception:
            pass
    except Exception:
        pass

    return templates.TemplateResponse(
        "case_detail.html",
        {
            "request": request,
            "case": case,
        },
    )

    if not case:
        return RedirectResponse(url="/cases", status_code=303)

    # fetch notes explicitly and attach so Jinja can access `case.notes`
    notes = db.query(Note).filter(Note.case_id == case_id).order_by(Note.id.desc()).all()
    try:
        setattr(case, "notes", notes)
    except Exception:
        pass

    offer = compute_offer_70(case.arv or 0, case.rehab or 0, case.closing_costs or 0)
    return templates.TemplateResponse(
        "case_detail.html",
        {
            "request": request,
            "case": case,
            "offer_70": offer,
            "active_parcel_id": case.parcel_id,
        },
    )

# ======================================================================
# SSE progress endpoints + update job orchestration
# ======================================================================
@app.get("/update_progress/{job_id}", response_class=HTMLResponse)
async def update_progress_page(request: Request, job_id: str):
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Updating cases…</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; }}
    .wrap {{ max-width: 900px; margin: 24px auto; padding: 0 16px; }}
    .spinner {{
      position: fixed; inset: 0; display: flex; align-items: center; justify-content: center;
      background: rgba(0,0,0,0.5); color: #fff; z-index: 9999; font-size: 18px;
    }}
    .log {{
      background: #0b0b0b; color: #c9f4ff; padding: 16px; border-radius: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      white-space: pre-wrap; line-height: 1.35; max-height: 60vh; overflow: auto;
      box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }}
    .muted {{ color: #9aa7b1; font-size: 12px; margin-top: 8px; }}
    .hide {{ display:none; }}
    .pill {{ display:inline-block; padding:4px 10px; border-radius: 999px; background:#eef2ff; color:#3730a3; font-size:12px; }}
  </style>
</head>
<body>
  <div id="spinner" class="spinner">Updating cases… Please don’t navigate away.</div>
  <div class="wrap">
    <h1>Update in progress <span class="pill">live log</span></h1>
    <div id="log" class="log"></div>
    <div id="hint" class="muted">This log will auto-scroll. You’ll be redirected when finished.</div>
  </div>

<script>
  const logEl = document.getElementById('log');
  const spinner = document.getElementById('spinner');
  const es = new EventSource('/events/{job_id}');
  function appendLine(s) {{
    logEl.textContent += s + '\\n';
    logEl.scrollTop = logEl.scrollHeight;
  }}
  es.onmessage = (e) => {{
    const t = e.data || '';
    if (t.startsWith('[done]')) {{
      spinner.classList.add('hide');
      es.close();
      setTimeout(() => window.location.href = '/cases', 10000);
    }} else {{
      if (t.trim().length) {{
        appendLine(t);
        spinner.classList.add('hide');
      }}
    }}
  }};
  es.onerror = () => {{
    appendLine('[connection error] retrying…');
  }};
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/events/{job_id}")
async def events(job_id: str):
    async def event_generator():
        # initial hello to open the stream promptly
        yield ": connected\n\n"
        while True:
            try:
                async for line in progress_bus.stream(job_id):
                    yield f"data: {line}\n\n"
            except Exception:
                # brief heartbeat to keep connection alive
                yield ": heartbeat\n\n"
                await asyncio.sleep(5)
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/import", response_class=HTMLResponse)
def update_case_list_page(request: Request):
    # Renders the form with the "Days to scrape" selector that posts to /update_cases
    return templates.TemplateResponse("import.html", {"request": request})


@app.post("/update_cases")
async def update_cases(
    request: Request,
    since_days: int = Form(7),
):
    """
    Starts an async job that:
      1) Runs the foreclosure scraper with --since-days
      2) Imports the resulting CSV with upsert-by-case_number (no dupes)
    Immediately redirects to a live log page.
    """
    job_id = uuid.uuid4().hex
    # prime the log so the progress page shows something immediately
    await progress_bus.publish(job_id, f"Queued job {job_id}…")
    asyncio.create_task(_update_cases_job(job_id, since_days))
    return RedirectResponse(url=f"/update_progress/{job_id}", status_code=303)


async def _update_cases_job(job_id: str, since_days: int):
    try:
        await progress_bus.publish(job_id, f"Starting update job {job_id} (since_days={since_days})")

        # 1) Run the scraper to produce CSV
        scraper_script = _find_scraper_script()
        tmpdir = tempfile.mkdtemp(prefix="pasco_update_")
        csv_out = os.path.join(tmpdir, "pasco_foreclosures.csv")

        cmd = [
            sys.executable,
            str(scraper_script),
            "--since-days", str(max(0, int(since_days))),
            "--out", csv_out,  # your integrated scraper should accept --out
        ]
        await progress_bus.publish(job_id, "Launching scraper: " + " ".join(cmd))
        rc = await run_command_with_logs(cmd, job_id)
        if rc != 0 or not os.path.exists(csv_out):
            await progress_bus.publish(job_id, "[error] Scraper failed or CSV not found.]")
            await progress_bus.publish(job_id, "[done] exit_code=1")
            return

        await progress_bus.publish(job_id, "Scraper finished. Importing CSV via tools/import_pasco_csv.py…")

        # 2) Import CSV using the same logic as the CLI tool
        def _run_import():
            # This is the same thing you just ran manually:
            # python tools/import_pasco_csv.py "C:\...\pasco_foreclosures.csv"
            import_pasco_csv_main(csv_out)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run_import)

        await progress_bus.publish(job_id, "Import complete via tools/import_pasco_csv.py")
        await progress_bus.publish(job_id, "[done] exit_code=0")

    except Exception as e:
        # Surface the exception in the log and signal completion
        await progress_bus.publish(job_id, f"[exception] {e}")
        await progress_bus.publish(job_id, "[done] exit_code=1")
# ======================================================================
# PDF Report for a Case (summary + attached documents)
# ======================================================================
@app.get("/cases/{case_id}/report")
def case_report(case_id: int, db: Session = Depends(get_db)):
    # Fetch case
    getter = getattr(db, "get", None)
    if callable(getter):
        case = db.get(Case, case_id)
    else:
        case = db.query(Case).get(case_id)  # type: ignore[call-arg]

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # -----------------------------
    # 1) Build summary/cover page
    # -----------------------------
    summary_buf = io.BytesIO()
    c = canvas.Canvas(summary_buf, pagesize=letter)
    width, height = letter

    y = height - 50

    def line(text: str, dy: int = 16, bold: bool = False):
        nonlocal y
        if bold:
            c.setFont("Helvetica-Bold", 12)
        else:
            c.setFont("Helvetica", 11)
        c.drawString(50, y, text)
        y -= dy

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "JSN Holdings - Case Report")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Case ID: {case.id}")
    y -= 18
    c.drawString(50, y, f"Case Number: {case.case_number or ''}")
    y -= 18

    # Basic info
    line(f"Filing Date: {case.filing_datetime or ''}")
    line(f"Style / Case Name: {case.style or ''}")
    line(f"Parcel ID: {case.parcel_id or ''}")
    addr = (getattr(case, 'address_override', None) or getattr(case, 'address', '') or '').strip()
    line(f"Address: {addr}")
    # Financials
    y -= 10
    line("Financials:", bold=True)
    arv_val = getattr(case, "arv", "") or ""
    rehab_val = getattr(case, "rehab", "") or ""
    closing_val = getattr(case, "closing_costs", "") or ""

    def _fmt_money(raw) -> str:
        # Accept str, int, float, or None and return a currency-formatted string when possible
        if raw is None:
            return ""
        # If it's already a number, format directly
        if isinstance(raw, (int, float)):
            try:
                return f"${float(raw):,.2f}"
            except Exception:
                return str(raw)
        raw_str = str(raw).strip()
        if not raw_str:
            return ""
        cleaned = raw_str.replace("$", "").replace(",", "")
        try:
            val = float(cleaned)
            return f"${val:,.2f}"
        except Exception:
            # fall back to original string if not parseable
            return raw_str

    line(f"ARV: {_fmt_money(arv_val)}")
    line(f"Rehab: {_fmt_money(rehab_val)}")
    line(f"Closing Costs: {_fmt_money(closing_val)}")

    # JSN deal calculator
    try:
        if isinstance(arv_val, (int, float)):
            arv_num = float(arv_val)
        else:
            arv_num = float((str(arv_val) or "").replace("$", "").replace(",", "") or 0)
    except Exception:
        arv_num = 0.0
    try:
        if isinstance(rehab_val, (int, float)):
            rehab_num = float(rehab_val)
        else:
            rehab_num = float((str(rehab_val) or "").replace("$", "").replace(",", "") or 0)
    except Exception:
        rehab_num = 0.0
    try:
        if isinstance(closing_val, (int, float)):
            closing_num = float(closing_val)
        else:
            closing_num = float((str(closing_val) or "").replace("$", "").replace(",", "") or 0)
    except Exception:
        closing_num = 0.0

    try:
        from app.utils import compute_offer_70
    except Exception:
        # Fallback: simple 70% ARV minus costs if helper missing
        def compute_offer_70(arv, rehab, closing):
            return max(0.0, 0.7 * (arv or 0.0) - (rehab or 0.0) - (closing or 0.0))

    offer_70 = compute_offer_70(arv_num, rehab_num, closing_num)
    try:
        offer_display = f"{offer_70:,.2f}"
    except Exception:
        offer_display = str(offer_70)
    line(f"JSN Max Offer: ${offer_display}")

    # Max Seller in Hand Cash (JSN Max Offer - sum of all liens)
    total_liens_for_calc = 0.0
    liens_raw_for_calc = getattr(case, "outstanding_liens", "[]") or "[]"
    try:
        liens_for_calc = json.loads(liens_raw_for_calc)
    except Exception:
        liens_for_calc = []
    if isinstance(liens_for_calc, list):
        for l in liens_for_calc:
            if isinstance(l, dict):
                amt_raw2 = (l.get("amount") or "").strip()
            else:
                amt_raw2 = ""
            if not amt_raw2:
                continue
            cleaned2 = amt_raw2.replace("$", "").replace(",", "")
            try:
                total_liens_for_calc += float(cleaned2)
            except Exception:
                continue
    try:
        seller_cash = max(0.0, float(offer_70) - float(total_liens_for_calc))
    except Exception:
        seller_cash = 0.0
    try:
        seller_display = f"{seller_cash:,.2f}"
    except Exception:
        seller_display = str(seller_cash)
    line(f"Max Seller in Hand Cash: ${seller_display}")

    # Defendants
    y -= 10
    line("Defendants:", bold=True)
    defendants = getattr(case, "defendants", []) or []
    clean_defendants = []
    for d in defendants:
        name = (getattr(d, "name", "") or "").strip()
        if not name:
            continue
        lower = name.lower()
        if lower in {"nan", "none", "nil"}:
            continue
        clean_defendants.append(name)
    if clean_defendants:
        for name in clean_defendants:
            line(f" - {name}")
    else:
        line(" - None recorded")
    # Outstanding Liens (if present)
    y -= 10
    line("Outstanding Liens:", bold=True)
    liens_raw = getattr(case, "outstanding_liens", "[]") or "[]"
    try:
        liens = json.loads(liens_raw)
    except Exception:
        liens = []
    total_liens = 0.0
    if isinstance(liens, list) and liens:
        for idx, l in enumerate(liens):
            if isinstance(l, dict):
                desc = (l.get("description") or "").strip()
                amt_raw = (l.get("amount") or "").strip()
            else:
                desc = str(l).strip()
                amt_raw = ""

            # Default description for first lien when missing → assume foreclosing mortgage
            if not desc:
                if idx == 0:
                    desc = "Foreclosing Mortgage"
                else:
                    desc = "Lien"

            # Normalize amount formatting
            amt_str = ""
            if amt_raw:
                cleaned = amt_raw.replace("$", "").replace(",", "")
                try:
                    total_liens += float(cleaned)
                except Exception:
                    pass
                # Pretty print
                try:
                    amt_val = float(cleaned)
                    amt_str = f"${amt_val:,.2f}"
                except Exception:
                    amt_str = f"${cleaned}"

            if amt_str:
                line(f" - {desc} - {amt_str}")
            else:
                line(f" - {desc}")
    else:
        line(" - None recorded")

    # Mortgage information (summarized from Mortgage PDF text if possible)
    mortgage_info = "Not uploaded"
    mort_rel = getattr(case, "mortgage_path", None)
    if mort_rel:
        try:
            mort_path = UPLOAD_ROOT / mort_rel
            if mort_path.exists():
                try:
                    r = PdfReader(str(mort_path))
                except Exception:
                    r = None
                if r is not None:
                    text_chunks = []
                    # read first 2 pages to keep it light
                    for page in r.pages[:2]:
                        try:
                            t = page.extract_text() or ""
                        except Exception:
                            t = ""
                        if t:
                            text_chunks.append(t)
                    full_text = " ".join(text_chunks)
                    full_text_norm = " ".join(full_text.split())
                    if full_text_norm:
                        snippet = full_text_norm[:200]
                        if len(full_text_norm) > 200:
                            snippet += "..."
                        mortgage_info = snippet
                    else:
                        mortgage_info = "Uploaded (no extractable text found)"
            else:
                mortgage_info = "Uploaded (file not found on disk)"
        except Exception:
            mortgage_info = "Uploaded (error reading file)"

    line(f"Mortgage Info: {mortgage_info}")

    # Notes summary (list recent notes)
    notes = getattr(case, "notes", None)
    if notes is None:
        # lazily load notes if not attached
        notes = db.query(Note).filter(Note.case_id == case_id).order_by(Note.id.desc()).all()
    y -= 10
    line("Notes:", bold=True)
    if notes:
        for n in notes:
            content = (getattr(n, "content", "") or "").strip()
            if not content:
                continue
            if len(content) > 160:
                content = content[:157] + "..."
            line(f" - {content}")
    else:
        line(" - None recorded")

    # Attached documents list
    y -= 10
    line("Attached Documents:", bold=True)
    attachments = []

    def add_doc(label: str, attr: str):
        rel = getattr(case, attr, None)
        if rel:
            attachments.append((label, rel))

    add_doc("Verified Complaint", "verified_complaint_path")
    add_doc("Value Calculation", "value_calc_path")
    add_doc("Mortgage", "mortgage_path")
    add_doc("Current Deed", "current_deed_path")
    add_doc("Previous Deed", "previous_deed_path")

    if attachments:
        for label, rel in attachments:
            line(f" - {label}: {rel}")
    else:
        line(" - No documents uploaded")

    c.showPage()
    c.save()
    summary_buf.seek(0)

    # -----------------------------
    # 2) Merge summary + attachments
    # -----------------------------
    writer = PdfWriter()

    # Add summary as first pages
    summary_reader = PdfReader(summary_buf)
    for page in summary_reader.pages:
        writer.add_page(page)

    # Now append each attached PDF if it exists on disk
    # Paths are stored relative to UPLOAD_ROOT, e.g. "CASE123/Verified_Complaint.pdf"
    from pathlib import Path as _Path

    for label, rel in attachments:
        # UPLOAD_ROOT is already defined in main.py
        abs_path = _Path(UPLOAD_ROOT) / rel
        if abs_path.exists():
            try:
                reader = PdfReader(str(abs_path))
                for page in reader.pages:
                    writer.add_page(page)
            except Exception:
                # Skip corrupted/unreadable PDF
                continue

    # Write combined PDF to buffer
    out_buf = io.BytesIO()
    writer.write(out_buf)
    out_buf.seek(0)

    filename = f"case_{case.id}_report.pdf"
    return StreamingResponse(
        out_buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

# ======================================================================
# Case editing / uploads / notes
# ======================================================================
@app.post("/cases/{case_id}/update")
def case_update(
    case_id: int,
    parcel_id: Optional[str] = Form(None),
    address_override: Optional[str] = Form(None),
    arv: Optional[float] = Form(None),
    rehab: Optional[float] = Form(None),
    closing_costs: Optional[float] = Form(None),
    db: Session = Depends(get_db),
):
    case = db.query(Case).get(case_id)  # type: ignore[call-arg]
    if not case:
        return RedirectResponse("/cases", status_code=303)
    if parcel_id is not None:
        case.parcel_id = (parcel_id or "").strip()
    if address_override is not None:
        case.address_override = (address_override or "").strip()
    if arv not in (None, ""):
        case.arv = float(arv)
    if rehab not in (None, ""):
        case.rehab = float(rehab)
    if closing_costs not in (None, ""):
        case.closing_costs = float(closing_costs)
    db.commit()
    return RedirectResponse(f"/cases/{case_id}", status_code=303)


@app.post("/cases/{case_id}/upload/verified")
async def upload_verified(
    case_id: int, verified_complaint: UploadFile = File(...), db: Session = Depends(get_db)
):
    case = db.query(Case).get(case_id)  # type: ignore[call-arg]
    if not case:
        return RedirectResponse("/cases", status_code=303)
    folder = ensure_case_folder(str(UPLOAD_ROOT), case.case_number)
    dest = Path(folder) / "Verified_Complaint.pdf"
    with open(dest, "wb") as f:
        f.write(await verified_complaint.read())
    case.verified_complaint_path = dest.relative_to(UPLOAD_ROOT).as_posix()
    db.commit()
    return RedirectResponse(f"/cases/{case_id}", status_code=303)


@app.post("/cases/{case_id}/upload/value_calc")
async def upload_value_calc(
    case_id: int, value_calc: UploadFile = File(...), db: Session = Depends(get_db)
):
    case = db.query(Case).get(case_id)  # type: ignore[call-arg]
    if not case:
        return RedirectResponse("/cases", status_code=303)
    folder = ensure_case_folder(str(UPLOAD_ROOT), case.case_number)
    dest = Path(folder) / "Value_Calculation.pdf"
    with open(dest, "wb") as f:
        f.write(await value_calc.read())
    case.value_calc_path = dest.relative_to(UPLOAD_ROOT).as_posix()
    db.commit()
    return RedirectResponse(f"/cases/{case_id}", status_code=303)


@app.post("/cases/{case_id}/upload/mortgage")
async def upload_mortgage(
    case_id: int, mortgage: UploadFile = File(...), db: Session = Depends(get_db)
):
    case = db.query(Case).get(case_id)  # type: ignore[call-arg]
    if not case:
        return RedirectResponse("/cases", status_code=303)
    folder = ensure_case_folder(str(UPLOAD_ROOT), case.case_number)
    dest = Path(folder) / "Mortgage.pdf"
    with open(dest, "wb") as f:
        f.write(await mortgage.read())
    case.mortgage_path = dest.relative_to(UPLOAD_ROOT).as_posix()
    db.commit()
    return RedirectResponse(f"/cases/{case_id}", status_code=303)


@app.post("/cases/{case_id}/upload/current-deed")
async def upload_current_deed(
    case_id: int, current_deed: UploadFile = File(...), db: Session = Depends(get_db)
):
    case = db.query(Case).get(case_id)  # type: ignore[call-arg]
    if not case:
        return RedirectResponse("/cases", status_code=303)
    folder = ensure_case_folder(str(UPLOAD_ROOT), case.case_number)
    dest = Path(folder) / "Current_Deed.pdf"
    with open(dest, "wb") as f:
        f.write(await current_deed.read())
    case.current_deed_path = dest.relative_to(UPLOAD_ROOT).as_posix()
    db.commit()
    return RedirectResponse(f"/cases/{case_id}", status_code=303)


@app.post("/cases/{case_id}/upload/previous-deed")
async def upload_previous_deed(
    case_id: int, previous_deed: UploadFile = File(...), db: Session = Depends(get_db)
):
    case = db.query(Case).get(case_id)  # type: ignore[call-arg]
    if not case:
        return RedirectResponse("/cases", status_code=303)
    folder = ensure_case_folder(str(UPLOAD_ROOT), case.case_number)
    dest = Path(folder) / "Previous_Deed.pdf"
    with open(dest, "wb") as f:
        f.write(await previous_deed.read())
    case.previous_deed_path = dest.relative_to(UPLOAD_ROOT).as_posix()
    db.commit()
    return RedirectResponse(f"/cases/{case_id}", status_code=303)


@app.post("/cases/{case_id}/notes/add")
def add_note(case_id: int, content: str = Form(...), db: Session = Depends(get_db)):
    case = db.query(Case).get(case_id)  # type: ignore[call-arg]
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    content = (content or "").strip()
    if not content:
        return RedirectResponse(url=f"/cases/{case_id}", status_code=303)
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    note = Note(case_id=case_id, content=content, created_at=ts)
    db.add(note)
    db.commit()
    return RedirectResponse(url=f"/cases/{case_id}", status_code=303)

# ======================================================================
# NEW: Outstanding Liens API
# ======================================================================
@app.get("/cases/{case_id}/liens", response_model=list[OutstandingLien])
def get_outstanding_liens(case_id: int, db: Session = Depends(get_db)):
    case = db.query(Case).get(case_id)  # type: ignore[call-arg]
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case.get_outstanding_liens()

@app.post("/cases/{case_id}/liens", response_model=list[OutstandingLien])
def save_outstanding_liens(case_id: int, payload: OutstandingLiensUpdate, db: Session = Depends(get_db)):
    case = db.query(Case).get(case_id)  # type: ignore[call-arg]
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    case.set_outstanding_liens([l.dict() for l in payload.outstanding_liens])
    db.add(case)
    db.commit()
    db.refresh(case)
    return case.get_outstanding_liens()


# ======================================================================
# Helpers: shell runner + scraper glue (robust Windows-friendly)
# ======================================================================
import asyncio, subprocess

async def run_command_with_logs(cmd: list[str], job_id: str) -> int:
    """
    Robust process runner that streams stdout -> progress_bus line by line.
    Uses asyncio subprocess when available; on Windows fallback to a thread.
    """
    # Try the native asyncio subprocess first
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            await progress_bus.publish(job_id, raw.decode(errors="ignore").rstrip("\n"))
        rc = await proc.wait()
        await progress_bus.publish(job_id, f"[done] exit_code={rc}")
        return rc

    except (NotImplementedError, RuntimeError, AttributeError):
        # Windows / event loop edge case: fallback to blocking Popen in a thread

        # Grab the current event loop here (in async context),
        # then use loop.call_soon_threadsafe() inside the worker thread.
        loop = asyncio.get_running_loop()

        def _blocking_runner() -> int:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            ) as p:
                assert p.stdout is not None
                for line in p.stdout:
                    # Hand off to the async loop via call_soon_threadsafe
                    loop.call_soon_threadsafe(
                        asyncio.create_task,
                        progress_bus.publish(job_id, line.rstrip("\n")),
                    )
                return p.wait()

        rc = await loop.run_in_executor(None, _blocking_runner)
        await progress_bus.publish(job_id, f"[done] exit_code={rc}")
        return rc



# Simple health check
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# =====================
# START: Added in v1.05+ for Archive + Export + Search
# =====================

# Ensure an 'archived' column exists (0/1) even if the ORM model lacks it
@app.on_event("startup")
def _ensure_archived_column():
    try:
        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("cases")}
        if "archived" not in cols:
            with engine.begin() as conn:
                conn.exec_driver_sql("ALTER TABLE cases ADD COLUMN archived INTEGER DEFAULT 0")
    except Exception as e:
        logger.warning("Could not ensure 'archived' column: %s", e)

@app.get("/cases", response_class=HTMLResponse)
def cases_list(
    request: Request,
    page: int = Query(1),
    show_archived: int = Query(0),
    case: str = Query("", alias="case"),
    db: Session = Depends(get_db),
):
    qry = db.query(Case)

    # Filter archived using a text where-clause (works even if ORM model lacks the column)
    if not show_archived:
        qry = qry.filter(text("(archived IS NULL OR archived = 0)"))

    if case:
        qry = qry.filter(Case.case_number.contains(case))

    page_size = 10
    total = qry.count()
    pages = (total + page_size - 1) // page_size
    offset = (page - 1) * page_size
    cases = (
        qry.order_by(Case.filing_datetime.desc())
           .offset(offset)
           .limit(page_size)
           .all()
    )
    pagination = {"page": page, "pages": pages, "total": total}
    return templates.TemplateResponse(
        "cases_list.html",
        {
            "request": request,
            "cases": cases,
            "pagination": pagination,
            "show_archived": bool(show_archived),
            "search_query": case,
        },
    )

@app.post("/cases/archive")
def archive_cases(
    request: Request,
    ids: List[int] = Form(default=[]),
    show_archived: int = Form(0),
    db: Session = Depends(get_db),
):
    if ids:
        # Raw SQL update that doesn't require Case.archived attribute on the ORM model
        db.execute(
            text("UPDATE cases SET archived = 1 WHERE id IN :ids")
            .bindparams(bindparam("ids", expanding=True)),
            {"ids": ids},
        )
        db.commit()
    # After archiving, force the list to hide archived items
    return RedirectResponse(url="/cases?show_archived=0&page=1", status_code=303)

@app.post("/cases/export")
def export_cases(
    request: Request,
    ids: List[int] = Form(default=[]),
    show_archived: int = Form(0),
    case: str = Form("", alias="case"),
    db: Session = Depends(get_db),
):
    qry = db.query(Case)

    # Mirror the list view’s archived filter via raw SQL text
    if not show_archived:
        qry = qry.filter(text("(archived IS NULL OR archived = 0)"))

    if case:
        qry = qry.filter(Case.case_number.contains(case))
    if ids:
        qry = qry.filter(Case.id.in_(ids))

    header = [
        "id",
        "case_number",
        "filing_datetime",
        "style",
        "address",
        "arv",
        "closing_costs",
        "current_deed_path",
        "defendants",
        "mortgage_path",
        "notes_count",
        "outstanding_liens",
        "parcel_id",
        "previous_deed_path",
        "rehab",
        "value_calc_path",
        "verified_complaint_path",
    ]

    buf = io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(header)

    rows = qry.order_by(Case.filing_datetime.desc()).all()
    for c in rows:
        try:
            defendants = [d.name for d in c.defendants] if getattr(c, "defendants", None) else []
        except Exception:
            defendants = []
        try:
            notes_count = len(c.notes) if getattr(c, "notes", None) else 0
        except Exception:
            notes_count = 0

        address = (getattr(c, "address_override", None) or getattr(c, "address", "") or "").strip()
        outstanding = getattr(c, "outstanding_liens", None) or "[]"

        writer.writerow([
            c.id,
            c.case_number or "",
            c.filing_datetime or "",
            c.style or "",
            address,
            getattr(c, "arv", "") or "",
            getattr(c, "closing_costs", "") or "",
            getattr(c, "current_deed_path", "") or "",
            json.dumps(defendants),
            getattr(c, "mortgage_path", "") or "",
            notes_count,
            outstanding,
            c.parcel_id or "",
            getattr(c, "previous_deed_path", "") or "",
            getattr(c, "rehab", "") or "",
            getattr(c, "value_calc_path", "") or "",
            getattr(c, "verified_complaint_path", "") or "",
        ])

    buf.seek(0)
    filename = f"cases_export_{_dt.datetime.now().strftime('%Y-%m-%d')}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )

# =====================
# END: Added in v1.05+
# =====================
# =====================
# v1.07 Additions — Unarchive + AJAX endpoints
# =====================

# Ensure 'archived' column exists (already in v1.06, keep for safety)
@app.on_event("startup")
def _ensure_archived_column_v107():
    try:
        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("cases")}
        if "archived" not in cols:
            with engine.begin() as conn:
                conn.exec_driver_sql("ALTER TABLE cases ADD COLUMN archived INTEGER DEFAULT 0")
    except Exception as e:
        logger.warning("Could not ensure 'archived' column: %s", e)

# --- Unarchive (form POST) ---
@app.post("/cases/unarchive")
def unarchive_cases(
    request: Request,
    ids: List[int] = Form(default=[]),
    show_archived: int = Form(0),
    db: Session = Depends(get_db),
):
    if ids:
        db.execute(
            text("UPDATE cases SET archived = 0 WHERE id IN :ids")
            .bindparams(bindparam("ids", expanding=True)),
            {"ids": ids},
        )
        db.commit()
    # After unarchiving, if user is showing archived, stay; else, reload hidden
    return RedirectResponse(url=f"/cases?show_archived={show_archived}&page=1", status_code=303)

# --- Archive (AJAX) ---
@app.post("/cases/archive_async")
def archive_cases_async(
    ids: List[int] = Form(default=[]),
    db: Session = Depends(get_db),
):
    if not ids:
        return {"ok": True, "updated": 0}
    db.execute(
        text("UPDATE cases SET archived = 1 WHERE id IN :ids")
        .bindparams(bindparam("ids", expanding=True)),
        {"ids": ids},
    )
    db.commit()
    return {"ok": True, "updated": len(ids)}

# --- Unarchive (AJAX) ---
@app.post("/cases/unarchive_async")
def unarchive_cases_async(
    ids: List[int] = Form(default=[]),
    db: Session = Depends(get_db),
):
    if not ids:
        return {"ok": True, "updated": 0}
    db.execute(
        text("UPDATE cases SET archived = 0 WHERE id IN :ids")
        .bindparams(bindparam("ids", expanding=True)),
        {"ids": ids},
    )
    db.commit()
    return {"ok": True, "updated": len(ids)}
# =====================
# Manual Add Case (v1.08)
# =====================



