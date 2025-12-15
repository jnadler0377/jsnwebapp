# app/services/update_cases_service.py

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import csv as _csv
from fastapi import HTTPException
from sqlalchemy.orm import Session
from datetime import datetime

from app.database import SessionLocal
from app.models import Case, Defendant
from app.services.progress_bus import progress_bus

logger = logging.getLogger("pascowebapp")

# Simple in-memory status for the last UpdateCases run.
# You can later surface this anywhere in the UI.
LAST_UPDATE_STATUS: Dict[str, Any] = {
    "last_run": None,     # ISO timestamp string
    "success": None,      # True / False / None
    "since_days": None,   # int
    "message": "",        # summary string
}


# Project root (same style as other services)
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # e.g. C:\pascowebapp


# ---------------------------------------------------------------------
# Subprocess helper – stream scraper output into the live log
# ---------------------------------------------------------------------
async def _stream_subprocess_to_progress(job_id: str, cmd: list[str]) -> int:
    """
    Run a subprocess, streaming stdout lines into the progress_bus.
    Returns the process exit code.
    """

async def _stream_subprocess_to_progress(job_id: str, cmd: list[str]) -> int:
    """
    Run a subprocess in a worker thread, streaming stdout into progress_bus.
    This avoids asyncio.create_subprocess_exec, which is problematic on
    Windows with the default event loop.
    """
    loop = asyncio.get_running_loop()

    def _run_and_stream() -> int:
        # Start the scraper process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,       # decode to str
            bufsize=1,       # line-buffered
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")

                # Schedule the publish coroutine safely back in the event loop
                loop.call_soon_threadsafe(
                    asyncio.create_task,
                    progress_bus.publish(job_id, line),
                )
        finally:
            # Ensure we wait for process exit and return its code
            rc = proc.wait()
        return rc

    # Run the blocking Popen/reading in a worker thread
    rc = await loop.run_in_executor(None, _run_and_stream)

    # Final status line
    await progress_bus.publish(job_id, f"[done] scraper_exit_code={rc}")
    return rc


# ---------------------------------------------------------------------
# Find the scraper script on disk
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# CSV import logic (upsert by normalized case_number)
# ---------------------------------------------------------------------
def _import_csv_into_db(db: Session, csv_path: str) -> Tuple[int, int, int]:
    """
    Lightweight importer (upsert by case_number using a normalized form).
    Returns (added, updated, skipped).
    """
    import re

    def norm_case(s) -> str:
        s = str(s or "").strip()
        # Normalize common separators away to reduce dupes
        s = s.replace("\\", "-").replace("/", "-")
        s = re.sub(r"\s+", "", s)
        return s

    def pick_col(headers, candidates):
        """
        Given headers and candidate names, return the first header that matches
        (case-insensitive, trimmed). Returns None if nothing matches.
        """
        norm_headers = [h.strip().lower() for h in headers if h]
        for cand in candidates:
            cand_norm = cand.strip().lower()
            if cand_norm in norm_headers:
                return headers[norm_headers.index(cand_norm)]
        return None

    added = updated = skipped = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f)
        headers = reader.fieldnames or []
        logger.info("UpdateCases: CSV headers = %s", headers)

        # Try multiple possible names for important columns
        case_col = pick_col(
            headers,
            ["Case #", "Case Number", "Case", "Case No.", "Case No", "CaseNum"],
        )
        filing_col = pick_col(headers, ["Filing Date", "Filing", "Filed"])
        style_col = pick_col(headers, ["Case Name", "Style", "Case Style"])
        addr_col = pick_col(
            headers,
            ["Property Address", "Address", "Site Address", "Situs Address"],
        )
        parcel_col = pick_col(
            headers,
            ["Parcel ID", "Folio", "Parcel", "ParcelID"],
        )

        if not case_col:
            msg = f"Could not find case number column in CSV headers: {headers}"
            logger.error("UpdateCases: %s", msg)
            raise HTTPException(status_code=400, detail=msg)

        # Build a lookup of existing cases by normalized case number
        existing_cases = db.query(Case).all()
        by_norm = {}
        for c in existing_cases:
            if c.case_number:
                by_norm[norm_case(c.case_number)] = c

        for row in reader:
            raw_case = row.get(case_col, "") or ""
            norm = norm_case(raw_case)
            if not norm:
                skipped += 1
                continue

            case = by_norm.get(norm)
            if not case:
                # New case
                case = Case(case_number=raw_case)
                if filing_col:
                    case.filing_datetime = row.get(filing_col, "") or None
                if style_col:
                    case.style = row.get(style_col, "") or None
                if addr_col:
                    case.address = row.get(addr_col, "") or None
                if parcel_col:
                    case.parcel_id = row.get(parcel_col, "") or None
                db.add(case)
                db.flush()  # assign id
                by_norm[norm] = case
                added += 1
            else:
                # Only fill blanks so we don't overwrite manual edits
                if not getattr(case, "filing_datetime", None) and filing_col:
                    case.filing_datetime = row.get(filing_col, "") or None
                if not getattr(case, "style", None) and style_col:
                    case.style = row.get(style_col, "") or None
                if not getattr(case, "address", None) and addr_col:
                    case.address = row.get(addr_col, "") or None
                if not getattr(case, "parcel_id", None) and parcel_col:
                    case.parcel_id = row.get(parcel_col, "") or None
                updated += 1

            # Defendants: any column whose header starts with "defendant"
            dnames = [
                (row.get(k, "") or "")
                for k in row.keys()
                if k and k.strip().lower().startswith("defendant")
            ]
            dnames = [d.strip() for d in dnames if d and d.strip()]

            existing_names = {d.name for d in (case.defendants or [])}
            for name in dnames:
                if name and name not in existing_names:
                    db.add(Defendant(case_id=case.id, name=name))
                    existing_names.add(name)

        db.commit()

    logger.info(
        "UpdateCases: Import complete. Added=%s Updated=%s Skipped=%s",
        added,
        updated,
        skipped,
    )
    return added, updated, skipped


# ---------------------------------------------------------------------
# Public orchestrator – called from main.py
# ---------------------------------------------------------------------
async def run_update_cases_job(job_id: str, since_days: int) -> None:
    """
    Orchestrator for the update job:
      1) Run the foreclosure scraper with --since-days
      2) Import the CSV results into the database
      3) Stream progress to the /progress SSE endpoint via progress_bus
    """
    started_at = datetime.utcnow().isoformat(timespec="seconds") 

    await progress_bus.publish(
        job_id,
        f"Starting update job {job_id} (since_days={since_days})",
    )

    tmpdir = tempfile.mkdtemp(prefix="pasco_update_")
    csv_out = os.path.join(tmpdir, "pasco_foreclosures.csv")

    try:
        scraper_script = _find_scraper_script()
        cmd = [
            sys.executable,
            str(scraper_script),
            "--since-days",
            str(max(0, int(since_days))),
            "--out",
            csv_out,
        ]

        await progress_bus.publish(job_id, f"Running scraper: {' '.join(cmd)}")
        rc = await _stream_subprocess_to_progress(job_id, cmd)
        if rc != 0:
            await progress_bus.publish(
                job_id,
                f"[error] Scraper exited with code {rc}, aborting import.",
            )
            await progress_bus.publish(job_id, "[done] exit_code=1")
            return

        # Import CSV into DB
        await progress_bus.publish(job_id, "Scraper finished, starting CSV import...")
        db: Session = SessionLocal()
        try:
            added, updated, skipped = _import_csv_into_db(db, csv_out)
        finally:
            db.close()

        # ✅ success: publish status + update LAST_UPDATE_STATUS
        msg = f"Import complete. added={added}, updated={updated}, skipped={skipped}"
        await progress_bus.publish(job_id, msg)

        LAST_UPDATE_STATUS.update(
            {
                "last_run": started_at,
                "success": True,
                "since_days": since_days,
                "message": msg,
            }
        )
        await progress_bus.publish(job_id, "[status] SUCCESS")
        await progress_bus.publish(job_id, "[done] exit_code=0")

    except Exception as exc:
        # ✅ failure: publish status + update LAST_UPDATE_STATUS
        logger.exception("UpdateCases: job failed: %s", exc)
        err_msg = str(exc)
        await progress_bus.publish(job_id, f"[exception] {err_msg}")

        LAST_UPDATE_STATUS.update(
            {
                "last_run": started_at,
                "success": False,
                "since_days": since_days,
                "message": err_msg,
            }
        )
        await progress_bus.publish(job_id, "[status] FAILURE")
        await progress_bus.publish(job_id, "[done] exit_code=1")
