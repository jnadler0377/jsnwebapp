
"""
Import Pasco foreclosure CSVs into the FastAPI app's SQLite DB.

Usage:
  python tools/import_pasco_csv.py /path/to/pasco_foreclosures.csv

Notes:
- Maps columns:
    "Case #" -> Case.case_number
    "Filing Date" -> Case.filing_datetime (string, original value)
    "Case Name" -> Case.style
    "Defendant 1..N" -> Defendant rows
- Upserts on case_number: if case exists, adds any missing defendants and updates filing/style if blank.
"""

import sys, os, re
import pandas as pd
from sqlalchemy.orm import Session
from pathlib import Path

# Make 'app' importable
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from app.database import Base, engine, SessionLocal
from app.models import Case, Defendant

def coalesce(*vals):
    for v in vals:
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""

def normalize_case_number(s):
    if not s:
        return ""
    s = str(s).strip()
    # Remove accidental whitespace and ensure consistent separators
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\", "-").replace("/", "-")
    return s

def main(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # Expected columns (case-insensitive)
    cols = {c.lower(): c for c in df.columns}

    def getcol(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_case = getcol("Case #", "Case#", "Case Number", "Case No")
    c_name = getcol("Case Name", "Style", "Case Style")
    c_file = getcol("Filing Date", "Filing", "Filed")

    defendant_cols = []
    for i in range(1, 20):
        c = getcol(f"Defendant {i}", f"Def {i}", f"Defendant{i}")
        if c:
            defendant_cols.append(c)

    if not c_case:
        print("Could not find a 'Case #' column. Aborting.")
        sys.exit(2)

    Base.metadata.create_all(bind=engine)
    session: Session = SessionLocal()
    created = 0
    updated = 0
    added_defendants = 0

    try:
        for _, row in df.iterrows():
            case_number = normalize_case_number(coalesce(row.get(c_case)))
            if not case_number:
                continue

            style = coalesce(row.get(c_name))
            filing = coalesce(row.get(c_file))

            case = session.query(Case).filter(Case.case_number == case_number).first()
            if not case:
                case = Case(case_number=case_number, style=style, filing_datetime=filing)
                session.add(case)
                session.flush()  # get id
                created += 1
            else:
                # Update any missing basic fields
                changed = False
                if style and (not case.style or case.style.strip() == ""):
                    case.style = style; changed = True
                if filing and (not case.filing_datetime or case.filing_datetime.strip() == ""):
                    case.filing_datetime = filing; changed = True
                if changed:
                    updated += 1

            # Add defendants
            existing = {d.name.strip().lower() for d in case.defendants if d.name}
            for dc in defendant_cols:
                name = coalesce(row.get(dc))
                if not name:
                    continue
                key = name.strip().lower()
                if key not in existing:
                    session.add(Defendant(case_id=case.id, name=name.strip()))
                    added_defendants += 1

        session.commit()
        print(f"Done. Cases created: {created}, cases updated: {updated}, defendants added: {added_defendants}")
    finally:
        session.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/import_pasco_csv.py /path/to/pasco_foreclosures.csv")
        sys.exit(64)
    main(sys.argv[1])
