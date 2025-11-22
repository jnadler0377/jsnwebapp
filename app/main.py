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
from sqlalchemy import inspect, text, bindparam, func  # add text, bindparam, func
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
            # Outstanding liens column (JSON stored as TEXT)
            if "outstanding_liens" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE cases ADD COLUMN outstanding_liens TEXT DEFAULT '[]'"
                )
            # Parsed mortgage helper columns
            if "mortgage_amount" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE cases ADD COLUMN mortgage_amount REAL DEFAULT 0"
                )
            if "mortgage_lender" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE cases ADD COLUMN mortgage_lender TEXT DEFAULT ''"
                )
            if "mortgage_borrower" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE cases ADD COLUMN mortgage_borrower TEXT DEFAULT ''"
                )
            if "mortgage_date" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE cases ADD COLUMN mortgage_date TEXT DEFAULT ''"
                )
            if "mortgage_recording_date" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE cases ADD COLUMN mortgage_recording_date TEXT DEFAULT ''"
                )
            if "mortgage_instrument" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE cases ADD COLUMN mortgage_instrument TEXT DEFAULT ''"
                )
            # Ensure archived column exists as INTEGER
            if "archived" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE cases ADD COLUMN archived INTEGER DEFAULT 0"
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

def _find_pinellas_scraper_script() -> Path:
    """
    Locate `pinellas_lis_pendens.py` in either:
    - <root>/app/integrations/pinellas_scraper
    - <root>/app/scrapers
    """
    candidates = [
        BASE_DIR / "app" / "integrations" / "pinellas_scraper" / "pinellas_lis_pendens.py",
        BASE_DIR / "app" / "scrapers" / "pinellas_lis_pendens.py",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Don't blow up the whole job; the caller can catch this and just log a warning
    raise HTTPException(
        status_code=500,
        detail=(
            "Pinellas scraper script not found. Place 'pinellas_lis_pendens.py' in "
            "'app/integrations/pinellas_scraper/' or 'app/scrapers/'."
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
    county_scope: str = Form("pasco"),
):
    """
    Starts an async job that:
      1) Runs one or both scrapers (Pasco / Pinellas) with --since-days
      2) Imports the Pasco CSV into the DB (Pinellas currently writes CSV only)
    Immediately redirects to a live log page.
    """
    # Normalize county scope
    county_scope = (county_scope or "pasco").strip().lower()
    if county_scope not in {"pasco", "pinellas", "both"}:
        county_scope = "pasco"

    job_id = uuid.uuid4().hex
    await progress_bus.publish(job_id, f"Queued job {job_id} (county_scope={county_scope})…")

    asyncio.create_task(_update_cases_job(job_id, since_days, county_scope))
    return RedirectResponse(url=f"/update_progress/{job_id}", status_code=303)



async def _update_cases_job(job_id: str, since_days: int, county_scope: str = "pasco"):
    """
    Background job that can:
      - Run the Pasco scraper and import into the DB
      - Run the Pinellas scraper and write a CSV
      - Or both, depending on county_scope
    """
    try:
        county_scope = (county_scope or "pasco").strip().lower()
        await progress_bus.publish(
            job_id,
            f"Starting update job {job_id} (since_days={since_days}, county_scope={county_scope})",
        )

        # -----------------------
        # 1) Pasco
        # -----------------------
        if county_scope in {"pasco", "both"}:
            await progress_bus.publish(job_id, "[pasco] Locating pasco scraper script…")
            scraper_script = _find_scraper_script()
            tmpdir = tempfile.mkdtemp(prefix="pasco_update_")
            csv_out = os.path.join(tmpdir, "pasco_foreclosures.csv")

            cmd = [
                sys.executable,
                str(scraper_script),
                "--since-days",
                str(max(0, int(since_days))),
                "--out",
                csv_out,
            ]
            await progress_bus.publish(job_id, "[pasco] Launching scraper: " + " ".join(cmd))
            rc = await run_command_with_logs(cmd, job_id)

            if rc != 0 or not os.path.exists(csv_out):
                await progress_bus.publish(
                    job_id,
                    "[pasco][error] Scraper failed or CSV not found.",
                )
                # For now, if Pasco fails and you requested Pasco or Both, bail out.
                if county_scope in {"pasco", "both"}:
                    await progress_bus.publish(job_id, "[done] exit_code=1")
                    return

            await progress_bus.publish(
                job_id,
                "[pasco] Scraper finished. Importing CSV via tools/import_pasco_csv.py…",
            )

            # Import CSV using the same logic as the CLI tool
            def _run_import():
                import_pasco_csv_main(csv_out)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _run_import)
            await progress_bus.publish(job_id, "[pasco] Import complete via tools/import_pasco_csv.py")

        # -----------------------
        # 2) Pinellas
        # -----------------------
        if county_scope in {"pinellas", "both"}:
            await progress_bus.publish(job_id, "[pinellas] Locating pinellas scraper script…")
            try:
                pinellas_script = _find_pinellas_scraper_script()
            except HTTPException as e:
                # Just log the problem instead of killing the whole job
                await progress_bus.publish(
                    job_id,
                    f"[pinellas][warning] {e.detail}",
                )
            else:
                # Write directly into the /data folder so you can inspect the CSV easily
                out_path = BASE_DIR / "data" / "pinellas_lis_pendens.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable,
                    str(pinellas_script),
                    "--since-days",
                    str(max(0, int(since_days))),
                    "--out",
                    str(out_path),
                ]
                await progress_bus.publish(job_id, "[pinellas] Launching scraper: " + " ".join(cmd))
                rc = await run_command_with_logs(cmd, job_id)

                if rc != 0:
                    await progress_bus.publish(
                        job_id,
                        "[pinellas][error] Scraper process exited with non-zero code.",
                    )
                else:
                    await progress_bus.publish(
                        job_id,
                        f"[pinellas] Scrape finished. CSV saved to: {out_path}",
                    )
                    await progress_bus.publish(
                        job_id,
                        "[pinellas] (Importer not yet wired; review CSV then we can add DB import.)",
                    )

        await progress_bus.publish(job_id, "[done] exit_code=0")

    except Exception as e:
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

    # Financials / Deal snapshot
    y -= 10
    line("Financials / Deal Snapshot:", bold=True)

    arv_val = getattr(case, "arv", 0) or 0
    rehab_val = getattr(case, "rehab", 0) or 0
    closing_val = getattr(case, "closing_costs", 0) or 0

    try:
        arv_val_f = float(arv_val)
    except Exception:
        arv_val_f = 0.0
    try:
        rehab_val_f = float(rehab_val)
    except Exception:
        rehab_val_f = 0.0
    try:
        closing_val_f = float(closing_val)
    except Exception:
        closing_val_f = 0.0

    offer_70 = compute_offer_70(arv_val_f, rehab_val_f, closing_val_f)

    # Sum outstanding liens (if any) as numbers
    liens_raw_for_calc = getattr(case, "outstanding_liens", "[]") or "[]"
    try:
        liens_for_calc = json.loads(liens_raw_for_calc)
    except Exception:
        liens_for_calc = []

    liens_total = 0.0
    if isinstance(liens_for_calc, list):
        for l in liens_for_calc:
            amt = None
            if isinstance(l, dict):
                amt = l.get("amount")
            else:
                amt = l
            if amt is None:
                continue
            try:
                import re as _re
                s = _re.sub(r"[^0-9.]+", "", str(amt))
                if s:
                    liens_total += float(s)
            except Exception:
                continue

    est_equity = max(0.0, arv_val_f - liens_total)
    max_wholesale_offer = max(0.0, offer_70 - liens_total)

    line(f"ARV: ${arv_val_f:,.0f}")
    line(f"Rehab: ${rehab_val_f:,.0f}")
    line(f"Closing Costs: ${closing_val_f:,.0f}")
    line(f"70% Offer (ARV - Rehab - Closing): ${offer_70:,.0f}")
    line(f"Total Outstanding Liens: ${liens_total:,.0f}")
    line(f"Estimated Equity (ARV - Liens): ${est_equity:,.0f}")
    line(f"Suggested Max Wholesale Offer: ${max_wholesale_offer:,.0f}")

    # Defendants
    y -= 10
    line("Defendants:", bold=True)
    defendants = getattr(case, "defendants", []) or []
    if defendants:
        for d in defendants:
            line(f" - {getattr(d, 'name', '')}")
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
    if isinstance(liens, list) and liens:
        for l in liens:
            desc = l.get("description") if isinstance(l, dict) else str(l)
            amt = l.get("amount") if isinstance(l, dict) else ""
            line(f" - {desc} {f'(${amt})' if amt else ''}")
    else:
        line(" - None recorded")

    # Notes summary (just show count)
    notes = getattr(case, "notes", None)
    if notes is None:
        # lazily load notes if not attached
        notes = db.query(Note).filter(Note.case_id == case_id).order_by(Note.id.desc()).all()
    y -= 10
    line(f"Notes Count: {len(notes) if notes else 0}", bold=True)

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


def extract_mortgage_fields_from_pdf(pdf_path: Path) -> dict:
    """Best-effort extraction of key mortgage fields from a PDF.

    Returns a dict with:
      mortgage_amount, mortgage_lender, mortgage_borrower,
      mortgage_date, mortgage_recording_date, mortgage_instrument
    All values may be None if not detected.
    """
    from PyPDF2 import PdfReader as _PdfReader
    import re as _re

    out = {
        "mortgage_amount": None,
        "mortgage_lender": None,
        "mortgage_borrower": None,
        "mortgage_date": None,
        "mortgage_recording_date": None,
        "mortgage_instrument": None,
    }

    try:
        reader = _PdfReader(str(pdf_path))
    except Exception:
        return out

    # Concatenate text from first few pages (mortgage docs are usually front-loaded)
    texts = []
    try:
        for i, page in enumerate(reader.pages[:5]):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            texts.append(t)
    except Exception:
        pass

    full = "\n".join(texts)
    if not full.strip():
        return out

    # Normalize spaces for easier regex
    norm = " ".join(full.split())

    # --- Amount: pick the largest dollar-looking number ---
    money_re = _re.compile(r"\$\s*([0-9]{2,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)")
    amounts = [m.group(1) for m in money_re.finditer(norm)]
    best_amt = None
    best_val = 0.0
    for amt in amounts:
        try:
            val = float(amt.replace(",", ""))
        except ValueError:
            continue
        # Heuristic: ignore tiny amounts
        if val >= best_val and val >= 1000:
            best_val = val
            best_amt = val
    if best_amt is not None:
        out["mortgage_amount"] = best_amt

    # --- Borrower / Mortgagor ---
    for label in ("Borrower:", "Mortgagor:", "GRANTOR:"):
        m = _re.search(label + r"\s*(.+?)(?:,|\n|\r|  )", full, flags=_re.IGNORECASE)
        if m:
            out["mortgage_borrower"] = m.group(1).strip()
            break

    # --- Lender / Mortgagee ---
    for label in ("Lender:", "Mortgagee:", "GRANTEE:"):
        m = _re.search(label + r"\s*(.+?)(?:,|\n|\r|  )", full, flags=_re.IGNORECASE)
        if m:
            out["mortgage_lender"] = m.group(1).strip()
            break

    # --- Dates: capture first couple of date-like patterns ---
    date_re = _re.compile(
        r"((?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d\d)"
    )
    dates = [m.group(1) for m in date_re.finditer(norm)]
    if dates:
        out["mortgage_date"] = dates[0]
        if len(dates) > 1:
            out["mortgage_recording_date"] = dates[1]

    # --- Instrument / Book & Page ---
    inst_re = _re.search(
        r"(Instrument\s+No\.\s*\d+|Book\s+\d+\s*,?\s*Page\s+\d+|OR\s+Book\s+\d+\s+Page\s+\d+)",
        full,
        flags=_re.IGNORECASE,
    )
    if inst_re:
        out["mortgage_instrument"] = inst_re.group(1).strip()

    return out



@app.get("/cases/{case_id}/deal-pdf")
def case_deal_pdf(case_id: int, db: Session = Depends(get_db)):
    """Export a single-page Deal Calculator PDF for this case."""
    getter = getattr(db, "get", None)
    if callable(getter):
        case = db.get(Case, case_id)
    else:
        case = db.query(Case).get(case_id)  # type: ignore[call-arg]

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    y = height - 50

    def line(text: str, dy: int = 18, bold: bool = False):
        nonlocal y
        if bold:
            c.setFont("Helvetica-Bold", 12)
        else:
            c.setFont("Helvetica", 11)
        c.drawString(50, y, text)
        y -= dy

    # Header
    c.setTitle(f"Deal Calculator - {case.case_number}")
    line("JSN Holdings - Deal Calculator", bold=True)
    y -= 4
    line(f"Case: {case.case_number}")
    line(f"Filed: {case.filing_datetime or ''}")
    addr = (getattr(case, "address_override", None) or getattr(case, "address", "") or "").strip()
    line(f"Address: {addr}")

    # Numbers
    y -= 10
    line("Numbers", bold=True)

    arv_val = getattr(case, "arv", 0) or 0
    rehab_val = getattr(case, "rehab", 0) or 0
    closing_val = getattr(case, "closing_costs", 0) or 0

    try:
        arv_val_f = float(arv_val)
    except Exception:
        arv_val_f = 0.0
    try:
        rehab_val_f = float(rehab_val)
    except Exception:
        rehab_val_f = 0.0
    try:
        closing_val_f = float(closing_val)
    except Exception:
        closing_val_f = 0.0

    offer_70 = compute_offer_70(arv_val_f, rehab_val_f, closing_val_f)

    liens_raw = getattr(case, "outstanding_liens", "[]") or "[]"
    try:
        liens_for_calc = json.loads(liens_raw)
    except Exception:
        liens_for_calc = []

    liens_total = 0.0
    if isinstance(liens_for_calc, list):
        for l in liens_for_calc:
            amt = None
            if isinstance(l, dict):
                amt = l.get("amount")
            else:
                amt = l
            if amt is None:
                continue
            try:
                import re as _re
                s = _re.sub(r"[^0-9.]+", "", str(amt))
                if s:
                    liens_total += float(s)
            except Exception:
                continue

    est_equity = max(0.0, arv_val_f - liens_total)
    max_wholesale_offer = max(0.0, offer_70 - liens_total)

    line(f"ARV: ${arv_val_f:,.0f}")
    line(f"Rehab: ${rehab_val_f:,.0f}")
    line(f"Closing Costs: ${closing_val_f:,.0f}")
    line(f"70% Offer (ARV - Rehab - Closing): ${offer_70:,.0f}")
    line(f"Total Outstanding Liens: ${liens_total:,.0f}")
    line(f"Estimated Equity (ARV - Liens): ${est_equity:,.0f}")
    line(f"Suggested Max Wholesale Offer: ${max_wholesale_offer:,.0f}")

    # Mortgage details if present
    y -= 10
    line("Mortgage (parsed)", bold=True)
    if getattr(case, "mortgage_amount", 0) or getattr(case, "mortgage_lender", ""):
        if getattr(case, "mortgage_amount", 0):
            try:
                line(f"Original Amount: ${float(case.mortgage_amount):,.0f}")
            except Exception:
                line(f"Original Amount: {case.mortgage_amount}")
        if getattr(case, "mortgage_lender", ""):
            line(f"Lender: {case.mortgage_lender}")
        if getattr(case, "mortgage_borrower", ""):
            line(f"Borrower: {case.mortgage_borrower}")
        if getattr(case, "mortgage_date", ""):
            line(f"Mortgage Date: {case.mortgage_date}")
        if getattr(case, "mortgage_recording_date", ""):
            line(f"Recording Date: {case.mortgage_recording_date}")
        if getattr(case, "mortgage_instrument", ""):
            line(f"Instrument: {case.mortgage_instrument}")
    else:
        line("No parsed mortgage details found.")

    c.showPage()
    c.save()

    buf.seek(0)
    filename = f"case_{case.id}_deal_calculator.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
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
    # Save relative path
    case.mortgage_path = dest.relative_to(UPLOAD_ROOT).as_posix()

    # Try to extract key mortgage fields from the PDF text
    try:
        fields = extract_mortgage_fields_from_pdf(dest)
    except Exception:
        fields = {}

    if isinstance(fields, dict):
        amt = fields.get("mortgage_amount")
        if amt is not None:
            try:
                case.mortgage_amount = float(amt)
            except Exception:
                pass
        lender = fields.get("mortgage_lender")
        if lender:
            case.mortgage_lender = str(lender)[:255]
        borrower = fields.get("mortgage_borrower")
        if borrower:
            case.mortgage_borrower = str(borrower)[:255]
        m_date = fields.get("mortgage_date")
        if m_date:
            case.mortgage_date = str(m_date)[:50]
        r_date = fields.get("mortgage_recording_date")
        if r_date:
            case.mortgage_recording_date = str(r_date)[:50]
        inst = fields.get("mortgage_instrument")
        if inst:
            case.mortgage_instrument = str(inst)[:255]

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
    min_arv: Optional[float] = Query(None),
    max_arv: Optional[float] = Query(None),
    has_mortgage: int = Query(0),
    has_current_deed: int = Query(0),
    has_liens: int = Query(0),
    has_notes: int = Query(0),
    filing_from: Optional[str] = Query(None),
    filing_to: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    qry = db.query(Case)

    # Filter archived using a text where-clause (works even if ORM model lacks the column)
    if not show_archived:
        qry = qry.filter(text("(archived IS NULL OR archived = 0)"))

    # Simple case-number search
    if case:
        qry = qry.filter(Case.case_number.contains(case))

    # Advanced numeric filters
    if min_arv is not None:
        qry = qry.filter(Case.arv >= min_arv)
    if max_arv is not None:
        qry = qry.filter(Case.arv <= max_arv)

    # Document presence filters
    if has_mortgage:
        qry = qry.filter(Case.mortgage_path.isnot(None), Case.mortgage_path != "")
    if has_current_deed:
        qry = qry.filter(Case.current_deed_path.isnot(None), Case.current_deed_path != "")
    if has_liens:
        # outstanding_liens stored as JSON TEXT; treat non-empty/non-[] as having liens
        qry = qry.filter(
            Case.outstanding_liens.isnot(None),
            Case.outstanding_liens != "",
            Case.outstanding_liens != "[]",
        )
    # Notes presence (via join + count)
    if has_notes:
        qry = qry.join(Note).group_by(Case.id).having(func.count(Note.id) > 0)

    # Filing date range (stored as YYYY-MM-DD string)
    if filing_from:
        qry = qry.filter(Case.filing_datetime >= filing_from)
    if filing_to:
        qry = qry.filter(Case.filing_datetime <= filing_to)

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
            "min_arv": min_arv,
            "max_arv": max_arv,
            "has_mortgage": has_mortgage,
            "has_current_deed": has_current_deed,
            "has_liens": has_liens,
            "has_notes": has_notes,
            "filing_from": filing_from,
            "filing_to": filing_to,
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



