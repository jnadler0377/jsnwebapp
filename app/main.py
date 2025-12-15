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
from tools.import_pasco_csv import main as import_pasco_csv_main
import requests  # for BatchData skip trace calls

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
from sqlalchemy import inspect, text, bindparam
from sqlalchemy.exc import OperationalError

# ---------------- App imports ----------------
from app.services.progress_bus import progress_bus
from app.settings import settings
from .database import Base, engine, SessionLocal
from .models import Case, Defendant, Docket, Note
from .utils import ensure_case_folder, compute_offer_70
from .schemas import OutstandingLien, OutstandingLiensUpdate
from dotenv import dotenv_values
from app.services.skiptrace_service import (
    get_case_address_components,
    batchdata_skip_trace,
    batchdata_property_lookup_all_attributes,
    save_property_for_case,
    save_skiptrace_row,
    load_property_for_case,
    load_skiptrace_for_case,
)

from app.services.report_service import generate_case_report
from app.services.update_cases_service import run_update_cases_job
from app.services.update_cases_service import LAST_UPDATE_STATUS


# Resolve project root (adjust if your .env lives somewhere else)
BASE_DIR = Path(__file__).resolve().parent.parent  # e.g. C:\pascowebapp
ENV_PATH = BASE_DIR / ".env"

# Read ONLY from the .env file
env_values = dotenv_values(ENV_PATH)

BATCHDATA_API_KEY = env_values.get("BATCHDATA_API_KEY")

print("DEBUG: .env path =", ENV_PATH)
print("DEBUG: BATCHDATA_API_KEY from .env =", BATCHDATA_API_KEY)

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


# ======================================================================
# BatchData Skip Trace config + helpers
# Helpers located app/services/skiptrace_service.py
# ======================================================================
BATCHDATA_API_KEY = env_values.get("BATCHDATA_API_KEY")
BATCHDATA_BASE_URL = "https://api.batchdata.com/api/v1"

templates.env.filters["currency"] = _currency
templates.env.globals["streetview_url"] = streetview_url
templates.env.globals["pasco_appraiser_url"] = pasco_appraiser_url

# ======================================================================
# DB session
# ======================================================================
def get_db():
    """
    Standard DB session dependency.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ======================================================================
# Skip Trace JSON Cache Helpers (legacy, still safe to keep)
# ======================================================================
def get_cached_skip_trace(case_id: int) -> Optional[dict]:
    """
    Read cached skip-trace JSON from the cases table, if any.
    """
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
                    "Failed to parse skip_trace_json for case %s: %s", case_id, exc
                )
                return None
    except Exception as exc:
        logger.warning(
            "Failed to read skip_trace_json for case %s: %s", case_id, exc
        )
    return None


def set_cached_skip_trace(case_id: int, payload: dict) -> None:
    """
    Persist skip-trace JSON into the cases.skip_trace_json column.
    """
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "UPDATE cases SET skip_trace_json = :payload WHERE id = :id",
                {"payload": json.dumps(payload), "id": case_id},
            )
    except Exception as exc:
        logger.warning(
            "Failed to write skip_trace_json for case %s: %s", case_id, exc
        )


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
            # Skip trace JSON cache
            if "skip_trace_json" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE cases ADD COLUMN skip_trace_json TEXT DEFAULT NULL"
                )
    except OperationalError:
        # first run or non-sqlite; ignore
        pass


# --------------------------------------------------------
#  SKIP TRACE NORMALIZED TABLE (CREATE ON STARTUP)
# --------------------------------------------------------
@app.on_event("startup")
# --------------------------------------------------------
#  SKIP TRACE NORMALIZED TABLES (CREATE ON STARTUP)
# --------------------------------------------------------
@app.on_event("startup")
def ensure_skiptrace_tables():
    """
    Ensure the skip-trace tables exist:

      - case_skiptrace         (1 row per case: owner + property address)
      - case_skiptrace_phone   (N rows per case: all phones)
      - case_skiptrace_email   (N rows per case: all emails)
    """
    try:
        with engine.begin() as conn:
            # Base summary table (leave existing extra columns alone if already created)
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS case_skiptrace (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER NOT NULL UNIQUE,
                    owner_name TEXT,
                    prop_street TEXT,
                    prop_city TEXT,
                    prop_state TEXT,
                    prop_zip TEXT,
                    FOREIGN KEY(case_id) REFERENCES cases(id)
                )
                """
            )

            # Phones: one row per phone record
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS case_skiptrace_phone (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER NOT NULL,
                    number TEXT,
                    type TEXT,
                    carrier TEXT,
                    last_reported TEXT,
                    score INTEGER,
                    tested INTEGER,
                    reachable INTEGER,
                    dnc INTEGER,
                    FOREIGN KEY(case_id) REFERENCES cases(id)
                )
                """
            )

            # Emails: one row per email record
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS case_skiptrace_email (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER NOT NULL,
                    email TEXT,
                    tested INTEGER,
                    FOREIGN KEY(case_id) REFERENCES cases(id)
                )
                """
            )
    except OperationalError:
        # sqlite / first run quirks; ignore
        pass
    except Exception as exc:
        logger.warning("Failed to ensure skip-trace tables: %s", exc)
# --------------------------------------------------------
#  PROPERTY LOOKUP TABLE (CREATE ON STARTUP)
# --------------------------------------------------------
# --------------------------------------------------------
#  PROPERTY DETAIL TABLE (CREATE/MIGRATE ON STARTUP)
# --------------------------------------------------------
@app.on_event("startup")
def ensure_property_table():
    """
    Ensure case_property exists with all expected columns.
    If the table already exists (older schema), add any missing columns.
    """
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        desired_ddl = """
            CREATE TABLE IF NOT EXISTS case_property (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL UNIQUE,

                -- BatchData property id
                batch_property_id TEXT,

                -- Address block
                address_validity        TEXT,
                address_house_number    TEXT,
                address_street          TEXT,
                address_city            TEXT,
                address_county          TEXT,
                address_state           TEXT,
                address_zip             TEXT,
                address_zip_plus4       TEXT,
                address_latitude        REAL,
                address_longitude       REAL,
                address_county_fips     TEXT,
                address_hash            TEXT,

                -- Demographics block
                demo_age                     INTEGER,
                demo_household_size          INTEGER,
                demo_income                  INTEGER,
                demo_net_worth               INTEGER,
                demo_discretionary_income    INTEGER,
                demo_homeowner_renter_code   TEXT,
                demo_homeowner_renter        TEXT,
                demo_gender_code             TEXT,
                demo_gender                  TEXT,
                demo_child_count             INTEGER,
                demo_has_children            INTEGER,
                demo_marital_status_code     TEXT,
                demo_marital_status          TEXT,
                demo_single_parent           INTEGER,
                demo_religious               INTEGER,
                demo_religious_affil_code    TEXT,
                demo_religious_affil         TEXT,
                demo_education_code          TEXT,
                demo_education               TEXT,
                demo_occupation              TEXT,
                demo_occupation_code         TEXT,

                -- Foreclosure block
                fc_status_code          TEXT,
                fc_status               TEXT,
                fc_recording_date       TEXT,
                fc_filing_date          TEXT,
                fc_case_number          TEXT,
                fc_auction_date         TEXT,
                fc_auction_time         TEXT,
                fc_auction_location     TEXT,
                fc_auction_city         TEXT,
                fc_auction_min_bid      REAL,
                fc_document_number      TEXT,
                fc_book_number          TEXT,
                fc_page_number          TEXT,
                fc_document_type_code   TEXT,
                fc_document_type        TEXT,

                -- Full deed history + full payload backup
                deed_history_json   TEXT,
                raw_json            TEXT,

                created_at          TEXT,
                updated_at          TEXT,

                FOREIGN KEY(case_id) REFERENCES cases(id)
            )
        """

        with engine.begin() as conn:
            # 1) Create table if it doesn't exist at all
            if "case_property" not in tables:
                conn.exec_driver_sql(desired_ddl)
                return

            # 2) If it DOES exist (older version), add missing columns
            existing_cols = {c["name"] for c in inspector.get_columns("case_property")}

            columns_to_add = [
                ("batch_property_id", "TEXT"),
                ("address_validity", "TEXT"),
                ("address_house_number", "TEXT"),
                ("address_street", "TEXT"),
                ("address_city", "TEXT"),
                ("address_county", "TEXT"),
                ("address_state", "TEXT"),
                ("address_zip", "TEXT"),
                ("address_zip_plus4", "TEXT"),
                ("address_latitude", "REAL"),
                ("address_longitude", "REAL"),
                ("address_county_fips", "TEXT"),
                ("address_hash", "TEXT"),
                ("demo_age", "INTEGER"),
                ("demo_household_size", "INTEGER"),
                ("demo_income", "INTEGER"),
                ("demo_net_worth", "INTEGER"),
                ("demo_discretionary_income", "INTEGER"),
                ("demo_homeowner_renter_code", "TEXT"),
                ("demo_homeowner_renter", "TEXT"),
                ("demo_gender_code", "TEXT"),
                ("demo_gender", "TEXT"),
                ("demo_child_count", "INTEGER"),
                ("demo_has_children", "INTEGER"),
                ("demo_marital_status_code", "TEXT"),
                ("demo_marital_status", "TEXT"),
                ("demo_single_parent", "INTEGER"),
                ("demo_religious", "INTEGER"),
                ("demo_religious_affil_code", "TEXT"),
                ("demo_religious_affil", "TEXT"),
                ("demo_education_code", "TEXT"),
                ("demo_education", "TEXT"),
                ("demo_occupation", "TEXT"),
                ("demo_occupation_code", "TEXT"),
                ("fc_status_code", "TEXT"),
                ("fc_status", "TEXT"),
                ("fc_recording_date", "TEXT"),
                ("fc_filing_date", "TEXT"),
                ("fc_case_number", "TEXT"),
                ("fc_auction_date", "TEXT"),
                ("fc_auction_time", "TEXT"),
                ("fc_auction_location", "TEXT"),
                ("fc_auction_city", "TEXT"),
                ("fc_auction_min_bid", "REAL"),
                ("fc_document_number", "TEXT"),
                ("fc_book_number", "TEXT"),
                ("fc_page_number", "TEXT"),
                ("fc_document_type_code", "TEXT"),
                ("fc_document_type", "TEXT"),
                ("deed_history_json", "TEXT"),
                ("raw_json", "TEXT"),
                ("created_at", "TEXT"),
                ("updated_at", "TEXT"),
            ]

            for col_name, col_type in columns_to_add:
                if col_name not in existing_cols:
                    conn.exec_driver_sql(
                        f"ALTER TABLE case_property ADD COLUMN {col_name} {col_type}"
                    )
    except Exception as exc:
        logger.warning("Failed to ensure/migrate case_property table: %s", exc)



# ======================================================================
# Helpers: shell runner + scraper glue
# ======================================================================


# ======================================================================
# Routes: home, list, detail
# ======================================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse(url="/cases", status_code=303)


@app.get("/cases/new", response_class=HTMLResponse)
def new_case_form(request: Request):
    return templates.TemplateResponse("cases_new.html", {"request": request, "error": None})

@app.get("/update_cases/status")
async def get_update_cases_status():
    """
    Returns the last UpdateCases job status:
      {
        "last_run": "2025-12-10T03:00:00",
        "success": true/false/null,
        "since_days": 1,
        "message": "Import complete. added=..., updated=..., skipped=..."
      }
    """
    return LAST_UPDATE_STATUS

@app.post("/cases/create")
def create_case(
    request: Request,
    case_number: str = Form(...),
    filing_date: Optional[str] = Form(None),   # "YYYY-MM-DD" or blank
    style: Optional[str] = Form(None),
    parcel_id: Optional[str] = Form(None),
    address_override: Optional[str] = Form(None),
    arv: Optional[str] = Form(None),
    rehab: Optional[str] = Form(None),
    closing_costs: Optional[str] = Form(None),
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
    if style:
        case.style = style.strip()
    if parcel_id:
        case.parcel_id = parcel_id.strip()
    if address_override:
        case.address_override = address_override.strip()

    # only set if provided
    v_arv = _num(arv)
    v_rehab = _num(rehab)
    v_cc = _num(closing_costs)
    if v_arv is not None:
        case.arv = v_arv
    if v_rehab is not None:
        case.rehab = v_rehab
    if v_cc is not None:
        case.closing_costs = v_cc

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

@app.post("/cases/{case_id}/update", response_class=HTMLResponse)
def update_case_fields(
    request: Request,
    case_id: int,
    parcel_id: Optional[str] = Form(None),
    address_override: Optional[str] = Form(None),
    arv: Optional[str] = Form(None),
    rehab: Optional[str] = Form(None),
    closing_costs: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    # Load case
    getter = getattr(db, "get", None)
    if callable(getter):
        case = db.get(Case, case_id)
    else:
        case = db.query(Case).get(case_id)  # type: ignore[call-arg]

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Helper to parse numbers (same logic as in create_case)
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

    # Text fields
    if parcel_id:
        case.parcel_id = parcel_id.strip()
    if address_override:
        case.address_override = address_override.strip()

    # Numbers (only set if provided)
    v_arv = _num(arv)
    v_rehab = _num(rehab)
    v_cc = _num(closing_costs)

    if v_arv is not None:
        case.arv = v_arv
    if v_rehab is not None:
        case.rehab = v_rehab
    if v_cc is not None:
        case.closing_costs = v_cc

    db.add(case)
    db.commit()

    # Send user back to the case detail page
    return RedirectResponse(
        url=request.url_for("case_detail", case_id=case.id),
        status_code=303,
    )

@app.get("/cases/{case_id}", response_class=HTMLResponse)
def case_detail(request: Request, case_id: int, db: Session = Depends(get_db)):
    getter = getattr(db, "get", None)
    if callable(getter):
        case = db.get(Case, case_id)
    else:
        case = db.query(Case).get(case_id)  # type: ignore[call-arg]

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    notes = (
        db.query(Note)
        .filter(Note.case_id == case_id)
        .order_by(Note.id.desc())
        .all()
    )
    try:
        setattr(case, "notes", notes)
    except Exception:
        pass

    # Skip trace from normalized table (if present)
    skip_trace = load_skiptrace_for_case(case_id)
    skip_trace_error = None

    # Property lookup (if present)
    property_data = load_property_for_case(case_id)
    property_error = None

    offer = compute_offer_70(case.arv or 0, case.rehab or 0, case.closing_costs or 0)

    return templates.TemplateResponse(
        "case_detail.html",
        {
            "request": request,
            "case": case,
            "offer_70": offer,
            "active_parcel_id": case.parcel_id,
            "notes": notes,
            "skip_trace": skip_trace,
            "skip_trace_error": skip_trace_error,
            "property_data": property_data,
            "property_error": property_error,
        },
    )



# NEW: Skip trace endpoint using BatchData
@app.post("/cases/{case_id}/skip-trace", response_class=HTMLResponse)
def skip_trace_case(request: Request, case_id: int, db: Session = Depends(get_db)):
    # Load case
    getter = getattr(db, "get", None)
    if callable(getter):
        case = db.get(Case, case_id)
    else:
        case = db.query(Case).get(case_id)  # type: ignore[call-arg]

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Notes
    notes = (
        db.query(Note)
        .filter(Note.case_id == case_id)
        .order_by(Note.id.desc())
        .all()
    )

    skip_trace: Optional[dict] = None
    skip_trace_error: Optional[str] = None

    # 1) Try table-based cache first
    skip_trace = load_skiptrace_for_case(case_id)

    # 2) If no stored data, call BatchData and persist normalized row
    if skip_trace is None:
        street, city, state, postal_code = get_case_address_components(case)

        try:
            skip_trace = batchdata_skip_trace(street, city, state, postal_code)
            # Save normalized into case_skiptrace
            save_skiptrace_row(case.id, skip_trace)
            # (optional) also keep JSON cache if you still want it:
            # set_cached_skip_trace(case_id, skip_trace)
        except HTTPException as exc:
            detail = exc.detail
            skip_trace_error = detail if isinstance(detail, str) else str(detail)
        except Exception as exc:
            skip_trace_error = f"Unexpected error during skip trace: {exc}"

    offer = compute_offer_70(case.arv or 0, case.rehab or 0, case.closing_costs or 0)

    return templates.TemplateResponse(
        "case_detail.html",
        {
            "request": request,
            "case": case,
            "offer_70": offer,
            "active_parcel_id": case.parcel_id,
            "notes": notes,
            "skip_trace": skip_trace,
            "skip_trace_error": skip_trace_error,
            "property_data": load_property_for_case(case_id),
            "property_error": None,
        },
    )

@app.post("/cases/{case_id}/property-lookup", response_class=HTMLResponse)
def property_lookup_case(request: Request, case_id: int, db: Session = Depends(get_db)):
    # Load case
    getter = getattr(db, "get", None)
    if callable(getter):
        case = db.get(Case, case_id)
    else:
        case = db.query(Case).get(case_id)  # type: ignore[call-arg]

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Notes
    notes = (
        db.query(Note)
        .filter(Note.case_id == case_id)
        .order_by(Note.id.desc())
        .all()
    )
    try:
        setattr(case, "notes", notes)
    except Exception:
        pass

    # Existing skip trace (unchanged)
    skip_trace = load_skiptrace_for_case(case_id)
    skip_trace_error: Optional[str] = None

    # Property lookup
    property_data: Optional[dict] = None
    property_error: Optional[str] = None

    try:
        street, city, state, postal_code = get_case_address_components(case)
        property_data = batchdata_property_lookup_all_attributes(
            street, city, state, postal_code
        )
        save_property_for_case(case.id, property_data)
    except HTTPException as exc:
        detail = exc.detail
        property_error = detail if isinstance(detail, str) else str(detail)
        # fall back to any previously saved data
        property_data = load_property_for_case(case_id)
    except Exception as exc:
        property_error = f"Unexpected error during property lookup: {exc}"
        property_data = load_property_for_case(case_id)

    offer = compute_offer_70(case.arv or 0, case.rehab or 0, case.closing_costs or 0)

    return templates.TemplateResponse(
        "case_detail.html",
        {
            "request": request,
            "case": case,
            "offer_70": offer,
            "active_parcel_id": case.parcel_id,
            "notes": notes,
            "skip_trace": skip_trace,
            "skip_trace_error": skip_trace_error,
            "property_data": property_data,
            "property_error": property_error,
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

    # delegate the heavy lifting to the service
    asyncio.create_task(run_update_cases_job(job_id, since_days))

    return RedirectResponse(
        url=request.url_for("update_progress_page", job_id=job_id),
        status_code=303,
    )



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
    """
    Lightweight wrapper that delegates to app.services.report_service.
    """
    return generate_case_report(case_id, db)


# ======================================================================
# Case document uploads
# ======================================================================
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

@app.post("/cases/{case_id}/documents/upload")
async def upload_case_document(
    case_id: int,
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Single upload endpoint. Uses doc_type to decide where to store the file:
    - verified          -> case.verified_complaint_path
    - mortgage          -> case.mortgage_path
    - current_deed      -> case.current_deed_path
    - previous_deed     -> case.previous_deed_path
    - value_calc        -> case.value_calc_path
    - other             -> generic Docket record
    """
    # Load case
    case = db.query(Case).get(case_id)  # type: ignore[call-arg]
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Normalize doc_type
    dt = (doc_type or "").strip().lower()

    # Make sure we have a filename
    original_name = file.filename or "document.pdf"
    safe_name = original_name.replace("/", "_").replace("\\", "_")

    # Folder per case
    folder = ensure_case_folder(str(UPLOAD_ROOT), case.case_number)

    # Map the dropdown choice to a fixed filename + case field
    mapping = {
        "verified":      ("Verified_Complaint.pdf", "verified_complaint_path", "Verified Complaint"),
        "mortgage":      ("Mortgage.pdf", "mortgage_path", "Mortgage"),
        "current_deed":  ("Current_Deed.pdf", "current_deed_path", "Current Deed"),
        "previous_deed": ("Previous_Deed.pdf", "previous_deed_path", "Previous Deed"),
        "value_calc":    ("Value_Calculation.pdf", "value_calc_path", "Value Calculation"),
    }

    if dt in mapping:
        target_name, attr_name, _label = mapping[dt]
        dest = Path(folder) / target_name
    else:
        # "other" or anything unknown: keep the user’s filename
        dest = Path(folder) / safe_name
        attr_name = None  # will create a Docket row instead

    # Save file to disk
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    rel_path = dest.relative_to(UPLOAD_ROOT).as_posix()

    # If it’s a known type, store on the Case model
    if attr_name:
        setattr(case, attr_name, rel_path)
        db.add(case)
        db.commit()
    else:
        # Generic "Other" document -> create a Docket record
        docket = Docket(
            case_id=case.id,
            file_name=safe_name,
            file_url=f"/uploads/{rel_path}",
            description=original_name,
        )
        db.add(docket)
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
# Simple health check
# ======================================================================
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# =====================
# START: Added in v1.05+ for Archive + Export + Search
# =====================
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
        db.execute(
            text("UPDATE cases SET archived = 1 WHERE id IN :ids")
            .bindparams(bindparam("ids", expanding=True)),
            {"ids": ids},
        )
        db.commit()
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
# v1.07 Additions — Unarchive + AJAX endpoints
# =====================
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
    return RedirectResponse(url=f"/cases?show_archived={show_archived}&page=1", status_code=303)


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
# (placeholder for future additions)
