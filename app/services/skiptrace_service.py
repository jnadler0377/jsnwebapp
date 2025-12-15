# app/services/skiptrace_service.py

from __future__ import annotations

import json
import logging
import datetime as _dt
from typing import Optional

import requests
from fastapi import HTTPException
from sqlalchemy import inspect

from app.database import engine
from app.settings import settings
from dotenv import dotenv_values
from pathlib import Path

logger = logging.getLogger("pascowebapp")

# ----------------------------------------------------------------------
# .env / BatchData config
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # adjust if needed
ENV_PATH = BASE_DIR / ".env"
env_values = dotenv_values(ENV_PATH)

BATCHDATA_API_KEY = env_values.get("BATCHDATA_API_KEY")
BATCHDATA_BASE_URL = "https://api.batchdata.com/api/v1"


# ----------------------------------------------------------------------
# Case address helpers
# ----------------------------------------------------------------------
def get_case_address_components(case) -> tuple[str, str, str, Optional[str]]:
    """
    Best-effort extraction of address components for BatchData skip trace.

    We try explicit fields first (if they exist on the model), otherwise
    we parse the combined address line into:
      street, city, state, postal_code
    """
    raw_addr = (getattr(case, "address_override", None) or getattr(case, "address", "") or "").strip()

    street = raw_addr
    city = ""
    state = "FL"
    postal_code: Optional[str] = None

    # If the ORM model has explicit fields, prefer them
    city_attr = getattr(case, "city", None)
    state_attr = getattr(case, "state", None)
    postal_attr = getattr(case, "postal_code", None) or getattr(case, "zip", None)

    if city_attr or state_attr or postal_attr:
        if city_attr:
            city = str(city_attr).strip()
        if state_attr:
            s = str(state_attr).strip()
            if s:
                state = s
        if postal_attr:
            postal_code = str(postal_attr).strip() or None
        if street:
            return street, city, state, postal_code

    # Fallback: parse from combined address string "123 Main St, City, ST 33556"
    if raw_addr and "," in raw_addr:
        parts = [p.strip() for p in raw_addr.split(",")]
        street = parts[0]
        if len(parts) >= 2:
            city = parts[1]
        if len(parts) >= 3:
            st_zip_parts = parts[2].split()
            if st_zip_parts:
                state = st_zip_parts[0]
            if len(st_zip_parts) > 1:
                postal_code = st_zip_parts[1]

    return street, city, state, postal_code


# ----------------------------------------------------------------------
# BatchData API calls
# ----------------------------------------------------------------------
def _require_key():
    if not BATCHDATA_API_KEY:
        raise HTTPException(status_code=500, detail="BatchData API key not configured")


def batchdata_skip_trace(
    street: str,
    city: str,
    state: str,
    postal_code: Optional[str] = None,
) -> dict:
    """
    Call BatchData Property Skip Trace API and normalize into a structure
    the template can use, including detailed phone/email metadata.
    """
    _require_key()

    if not street or not city or not state:
        raise HTTPException(status_code=400, detail="Incomplete address for skip trace")

    url = f"{BATCHDATA_BASE_URL}/property/skip-trace"

    property_address: dict = {"street": street, "city": city, "state": state}
    if postal_code:
        property_address["postalCode"] = postal_code

    payload = {"requests": [{"propertyAddress": property_address}]}

    headers = {
        "Authorization": f"Bearer {BATCHDATA_API_KEY}",
        "Content-Type": "application/json",
    }

    # Log (masked)
    masked_headers = headers.copy()
    if "Authorization" in masked_headers:
        token = masked_headers["Authorization"]
        if len(token) > 20:
            masked_headers["Authorization"] = token[:20] + "...(masked)"

    print("\n\n=================== BATCHDATA REQUEST ===================")
    print("URL:", url)
    print("HEADERS:", masked_headers)
    print("PAYLOAD:", json.dumps(payload, indent=2))
    print("=========================================================\n")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
    except Exception as exc:
        print("\n\n=================== BATCHDATA ERROR =====================")
        print(exc)
        print("=========================================================\n")
        raise HTTPException(status_code=502, detail=f"Error calling BatchData API: {exc}")

    print("\n\n================== BATCHDATA RESPONSE ====================")
    print("STATUS:", resp.status_code)
    print("TEXT:", resp.text[:5000])
    print("==========================================================\n")

    if resp.status_code >= 400:
        try:
            err_json = resp.json()
        except Exception:
            err_json = resp.text

        if resp.status_code == 403:
            raise HTTPException(
                status_code=403,
                detail=(
                    "BatchData: this API key does not have permission for the "
                    "Property Skip Trace endpoint. Check your BatchData plan or API key settings."
                ),
            )

        raise HTTPException(status_code=resp.status_code, detail=f"BatchData error: {err_json}")

    try:
        data = resp.json()
    except Exception:
        raise HTTPException(
            status_code=502,
            detail=f"BatchData returned non-JSON response: {resp.text[:500]}",
        )

    # Normalization
    out_results: list[dict] = []

    if isinstance(data, dict):
        res = data.get("results")
        if isinstance(res, dict):
            raw_results = [res]
        elif isinstance(res, list):
            raw_results = res
        else:
            resp_obj = data.get("response")
            if isinstance(resp_obj, dict) and isinstance(resp_obj.get("results"), list):
                raw_results = resp_obj["results"]
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Unexpected BatchData response structure (dict, no usable results): {data}",
                )
    elif isinstance(data, list):
        raw_results = data
    else:
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected BatchData response type: {type(data).__name__} -> {data!r}",
        )

    for r in raw_results:
        if not isinstance(r, dict):
            continue

        persons_raw = r.get("persons") or []
        if not isinstance(persons_raw, list):
            persons_raw = []

        simple_persons: list[dict] = []
        property_addr_result: dict = r.get("propertyAddress") or {}

        for p in persons_raw:
            if not isinstance(p, dict):
                continue

            # property address
            if not property_addr_result:
                pa = p.get("propertyAddress")
                if not isinstance(pa, dict):
                    prop = p.get("property") or {}
                    if isinstance(prop, dict):
                        pa = prop.get("address")
                if isinstance(pa, dict):
                    property_addr_result = pa

            # name
            full_name = ""
            if isinstance(p.get("fullName"), str):
                full_name = p["fullName"]
            else:
                name_obj = p.get("name")
                if isinstance(name_obj, dict):
                    full_name = (
                        name_obj.get("full")
                        or " ".join(
                            x for x in [name_obj.get("first"), name_obj.get("last")] if x
                        )
                    )
                elif isinstance(name_obj, str):
                    full_name = name_obj

            # emails
            emails: list[dict] = []
            emails_raw = p.get("emails") or []
            if isinstance(emails_raw, list):
                for e in emails_raw:
                    if isinstance(e, dict):
                        email = e.get("email")
                        tested = e.get("tested")
                        if email:
                            emails.append(
                                {
                                    "email": email,
                                    "tested": bool(tested) if isinstance(tested, bool) else None,
                                }
                            )
                    elif isinstance(e, str):
                        emails.append({"email": e, "tested": None})

            # phones
            phones: list[dict] = []
            phones_raw = p.get("phoneNumbers") or []
            if isinstance(phones_raw, list):
                for ph in phones_raw:
                    if not isinstance(ph, dict):
                        continue
                    number = ph.get("number") or ph.get("phone")
                    if not number:
                        continue
                    phone_type = ph.get("type")
                    carrier = ph.get("carrier")
                    tested = ph.get("tested")
                    reachable = ph.get("reachable")
                    dnc = ph.get("dnc")
                    last_reported = ph.get("lastReportedDate")
                    score = ph.get("score")
                    phones.append(
                        {
                            "number": number,
                            "type": phone_type,
                            "carrier": carrier,
                            "tested": bool(tested) if isinstance(tested, bool) else None,
                            "reachable": bool(reachable) if isinstance(reachable, bool) else None,
                            "dnc": bool(dnc) if isinstance(dnc, bool) else None,
                            "last_reported": last_reported,
                            "score": score,
                        }
                    )

            simple_persons.append(
                {
                    "full_name": full_name,
                    "emails": emails,
                    "phones": phones,
                }
            )

        out_results.append(
            {
                "propertyAddress": property_addr_result or {},
                "persons": simple_persons,
            }
        )

    return {"results": out_results}


def batchdata_property_lookup_all_attributes(
    street: str,
    city: str,
    state: str,
    postal_code: Optional[str] = None,
) -> dict:
    """
    Call BatchData Property Lookup (all-attributes) endpoint and
    return the raw JSON payload.
    """
    _require_key()

    if not street or not city or not state:
        raise HTTPException(status_code=400, detail="Incomplete address for property lookup")

    url = f"{BATCHDATA_BASE_URL}/property/lookup/all-attributes"

    property_address: dict = {"street": street, "city": city, "state": state}
    if postal_code:
        property_address["postalCode"] = postal_code

    payload = {"requests": [{"address": property_address}]}

    headers = {
        "Authorization": f"Bearer {BATCHDATA_API_KEY}",
        "Content-Type": "application/json",
    }

    masked_headers = headers.copy()
    if "Authorization" in masked_headers:
        token = masked_headers["Authorization"]
        if len(token) > 20:
            masked_headers["Authorization"] = token[:20] + "...(masked)"

    print("\n\n=================== BATCHDATA PROPERTY LOOKUP ===================")
    print("URL:", url)
    print("HEADERS:", masked_headers)
    print("PAYLOAD:", json.dumps(payload, indent=2))
    print("===============================================================\n")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
    except Exception as exc:
        print("BatchData property lookup ERROR:", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Error calling BatchData Property Lookup API: {exc}",
        )

    print("\n\n================== BATCHDATA PROPERTY RESPONSE ==================")
    print("STATUS:", resp.status_code)
    print("TEXT:", resp.text[:5000])
    print("===============================================================\n")

    if resp.status_code >= 400:
        try:
            err_json = resp.json()
        except Exception:
            err_json = resp.text
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"BatchData property lookup error: {err_json}",
        )

    try:
        return resp.json()
    except Exception:
        raise HTTPException(
            status_code=502,
            detail=f"BatchData returned non-JSON response: {resp.text[:500]}",
        )


# ----------------------------------------------------------------------
# DB helpers: property + skiptrace tables
# ----------------------------------------------------------------------
def save_property_for_case(case_id: int, payload: dict) -> None:
    """
    Upsert a single row in case_property for this case_id.
    Maps the first property in payload['results']['properties'] into columns.
    """
    def b2i(val):
        if isinstance(val, bool):
            return 1 if val else 0
        return None

    results = (payload or {}).get("results") or {}
    props = results.get("properties") or []
    if not props:
        logger.warning("save_property_for_case: no 'properties' in payload for case %s", case_id)
        return

    prop = props[0]  # use first result
    address = prop.get("address") or {}
    demo = prop.get("demographics") or {}
    fc = prop.get("foreclosure") or {}
    deed_history = prop.get("deedHistory") or []

    ts = _dt.datetime.utcnow().isoformat()

    vals = {
        "case_id": case_id,
        "batch_property_id": prop.get("_id"),
        "address_validity":    address.get("addressValidity"),
        "address_house_number": address.get("houseNumber"),
        "address_street":      address.get("street"),
        "address_city":        address.get("city"),
        "address_county":      address.get("county"),
        "address_state":       address.get("state"),
        "address_zip":         address.get("zip"),
        "address_zip_plus4":   address.get("zipPlus4"),
        "address_latitude":    address.get("latitude"),
        "address_longitude":   address.get("longitude"),
        "address_county_fips": address.get("countyFipsCode"),
        "address_hash":        address.get("hash"),
        "demo_age":                   demo.get("age"),
        "demo_household_size":        demo.get("householdSize"),
        "demo_income":                demo.get("income"),
        "demo_net_worth":            demo.get("netWorth"),
        "demo_discretionary_income":  demo.get("discretionaryIncome"),
        "demo_homeowner_renter_code": demo.get("homeownerRenterCode"),
        "demo_homeowner_renter":      demo.get("homeownerRenter"),
        "demo_gender_code":           demo.get("genderCode"),
        "demo_gender":                demo.get("gender"),
        "demo_child_count":           demo.get("childCount"),
        "demo_has_children":          b2i(demo.get("hasChildren")),
        "demo_marital_status_code":   demo.get("maritalStatusCode"),
        "demo_marital_status":        demo.get("maritalStatus"),
        "demo_single_parent":         b2i(demo.get("singleParent")),
        "demo_religious":             b2i(demo.get("religious")),
        "demo_religious_affil_code":  demo.get("religiousAffiliationCode"),
        "demo_religious_affil":       demo.get("religiousAffiliation"),
        "demo_education_code":        demo.get("individualEducationCode"),
        "demo_education":             demo.get("individualEducation"),
        "demo_occupation":            demo.get("individualOccupation"),
        "demo_occupation_code":       demo.get("individualOccupationCode"),
        "fc_status_code":       fc.get("statusCode"),
        "fc_status":            fc.get("status"),
        "fc_recording_date":    fc.get("recordingDate"),
        "fc_filing_date":       fc.get("filingDate"),
        "fc_case_number":       fc.get("caseNumber"),
        "fc_auction_date":      fc.get("auctionDate"),
        "fc_auction_time":      fc.get("auctionTime"),
        "fc_auction_location":  fc.get("auctionLocation"),
        "fc_auction_city":      fc.get("auctionCity"),
        "fc_auction_min_bid":   fc.get("auctionMinimumBidAmount"),
        "fc_document_number":   fc.get("documentNumber"),
        "fc_book_number":       fc.get("bookNumber"),
        "fc_page_number":       fc.get("pageNumber"),
        "fc_document_type_code": fc.get("documentTypeCode"),
        "fc_document_type":     fc.get("documentType"),
        "deed_history_json": json.dumps(deed_history) if deed_history else None,
        "raw_json":          json.dumps(payload),
        "created_at": ts,
        "updated_at": ts,
    }

    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                """
                INSERT INTO case_property (
                    case_id,
                    batch_property_id,
                    address_validity,
                    address_house_number,
                    address_street,
                    address_city,
                    address_county,
                    address_state,
                    address_zip,
                    address_zip_plus4,
                    address_latitude,
                    address_longitude,
                    address_county_fips,
                    address_hash,
                    demo_age,
                    demo_household_size,
                    demo_income,
                    demo_net_worth,
                    demo_discretionary_income,
                    demo_homeowner_renter_code,
                    demo_homeowner_renter,
                    demo_gender_code,
                    demo_gender,
                    demo_child_count,
                    demo_has_children,
                    demo_marital_status_code,
                    demo_marital_status,
                    demo_single_parent,
                    demo_religious,
                    demo_religious_affil_code,
                    demo_religious_affil,
                    demo_education_code,
                    demo_education,
                    demo_occupation,
                    demo_occupation_code,
                    fc_status_code,
                    fc_status,
                    fc_recording_date,
                    fc_filing_date,
                    fc_case_number,
                    fc_auction_date,
                    fc_auction_time,
                    fc_auction_location,
                    fc_auction_city,
                    fc_auction_min_bid,
                    fc_document_number,
                    fc_book_number,
                    fc_page_number,
                    fc_document_type_code,
                    fc_document_type,
                    deed_history_json,
                    raw_json,
                    created_at,
                    updated_at
                )
                VALUES (
                    :case_id,
                    :batch_property_id,
                    :address_validity,
                    :address_house_number,
                    :address_street,
                    :address_city,
                    :address_county,
                    :address_state,
                    :address_zip,
                    :address_zip_plus4,
                    :address_latitude,
                    :address_longitude,
                    :address_county_fips,
                    :address_hash,
                    :demo_age,
                    :demo_household_size,
                    :demo_income,
                    :demo_net_worth,
                    :demo_discretionary_income,
                    :demo_homeowner_renter_code,
                    :demo_homeowner_renter,
                    :demo_gender_code,
                    :demo_gender,
                    :demo_child_count,
                    :demo_has_children,
                    :demo_marital_status_code,
                    :demo_marital_status,
                    :demo_single_parent,
                    :demo_religious,
                    :demo_religious_affil_code,
                    :demo_religious_affil,
                    :demo_education_code,
                    :demo_education,
                    :demo_occupation,
                    :demo_occupation_code,
                    :fc_status_code,
                    :fc_status,
                    :fc_recording_date,
                    :fc_filing_date,
                    :fc_case_number,
                    :fc_auction_date,
                    :fc_auction_time,
                    :fc_auction_location,
                    :fc_auction_city,
                    :fc_auction_min_bid,
                    :fc_document_number,
                    :fc_book_number,
                    :fc_page_number,
                    :fc_document_type_code,
                    :fc_document_type,
                    :deed_history_json,
                    :raw_json,
                    :created_at,
                    :updated_at
                )
                ON CONFLICT(case_id) DO UPDATE SET
                    batch_property_id          = excluded.batch_property_id,
                    address_validity           = excluded.address_validity,
                    address_house_number       = excluded.address_house_number,
                    address_street             = excluded.address_street,
                    address_city               = excluded.address_city,
                    address_county             = excluded.address_county,
                    address_state              = excluded.address_state,
                    address_zip                = excluded.address_zip,
                    address_zip_plus4          = excluded.address_zip_plus4,
                    address_latitude           = excluded.address_latitude,
                    address_longitude          = excluded.address_longitude,
                    address_county_fips        = excluded.address_county_fips,
                    address_hash               = excluded.address_hash,
                    demo_age                   = excluded.demo_age,
                    demo_household_size        = excluded.demo_household_size,
                    demo_income                = excluded.demo_income,
                    demo_net_worth             = excluded.demo_net_worth,
                    demo_discretionary_income  = excluded.demo_discretionary_income,
                    demo_homeowner_renter_code = excluded.demo_homeowner_renter_code,
                    demo_homeowner_renter      = excluded.demo_homeowner_renter,
                    demo_gender_code           = excluded.demo_gender_code,
                    demo_gender                = excluded.demo_gender,
                    demo_child_count           = excluded.demo_child_count,
                    demo_has_children          = excluded.demo_has_children,
                    demo_marital_status_code   = excluded.demo_marital_status_code,
                    demo_marital_status        = excluded.demo_marital_status,
                    demo_single_parent         = excluded.demo_single_parent,
                    demo_religious             = excluded.demo_religious,
                    demo_religious_affil_code  = excluded.demo_religious_affil_code,
                    demo_religious_affil       = excluded.demo_religious_affil,
                    demo_education_code        = excluded.demo_education_code,
                    demo_education             = excluded.demo_education,
                    demo_occupation            = excluded.demo_occupation,
                    demo_occupation_code       = excluded.demo_occupation_code,
                    fc_status_code             = excluded.fc_status_code,
                    fc_status                  = excluded.fc_status,
                    fc_recording_date          = excluded.fc_recording_date,
                    fc_filing_date             = excluded.fc_filing_date,
                    fc_case_number             = excluded.fc_case_number,
                    fc_auction_date            = excluded.fc_auction_date,
                    fc_auction_time            = excluded.fc_auction_time,
                    fc_auction_location        = excluded.fc_auction_location,
                    fc_auction_city            = excluded.fc_auction_city,
                    fc_auction_min_bid         = excluded.fc_auction_min_bid,
                    fc_document_number         = excluded.fc_document_number,
                    fc_book_number             = excluded.fc_book_number,
                    fc_page_number             = excluded.fc_page_number,
                    fc_document_type_code      = excluded.fc_document_type_code,
                    fc_document_type           = excluded.fc_document_type,
                    deed_history_json          = excluded.deed_history_json,
                    raw_json                   = excluded.raw_json,
                    updated_at                 = excluded.updated_at
                """,
                vals,
            )
    except Exception as exc:
        logger.warning("Failed to save property lookup for case %s: %s", case_id, exc)


def save_skiptrace_row(case_id: int, skip_trace: dict) -> None:
    """
    Take our normalized skip_trace dict (from batchdata_skip_trace) and
    persist ALL phones/emails into case_skiptrace_phone / case_skiptrace_email,
    plus a summary row in case_skiptrace.
    """
    try:
        results = (skip_trace or {}).get("results") or []
        if not results:
            return

        res = results[0]
        persons = res.get("persons") or []
        if not persons:
            return

        p = persons[0]

        owner_name = p.get("full_name") or ""

        prop_addr = res.get("propertyAddress") or {}
        prop_street = prop_addr.get("street")
        prop_city = prop_addr.get("city")
        prop_state = prop_addr.get("state")
        prop_zip = prop_addr.get("zip") or prop_addr.get("postalCode")

        phones = p.get("phones") or []
        emails = p.get("emails") or []

        with engine.begin() as conn:
            conn.exec_driver_sql(
                """
                INSERT INTO case_skiptrace (
                    case_id,
                    owner_name,
                    prop_street, prop_city, prop_state, prop_zip
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    owner_name = excluded.owner_name,
                    prop_street = excluded.prop_street,
                    prop_city = excluded.prop_city,
                    prop_state = excluded.prop_state,
                    prop_zip = excluded.prop_zip
                """,
                (case_id, owner_name, prop_street, prop_city, prop_state, prop_zip),
            )

            conn.exec_driver_sql("DELETE FROM case_skiptrace_phone WHERE case_id = ?", (case_id,))
            conn.exec_driver_sql("DELETE FROM case_skiptrace_email WHERE case_id = ?", (case_id,))

            def as_int_bool(val):
                if isinstance(val, bool):
                    return int(val)
                return None

            for ph in phones:
                if not isinstance(ph, dict):
                    continue
                number = ph.get("number")
                if not number:
                    continue
                phone_type = ph.get("type")
                carrier = ph.get("carrier")
                last_reported = ph.get("last_reported") or ph.get("lastReportedDate")
                score = ph.get("score")
                tested = as_int_bool(ph.get("tested"))
                reachable = as_int_bool(ph.get("reachable"))
                dnc = as_int_bool(ph.get("dnc"))

                conn.exec_driver_sql(
                    """
                    INSERT INTO case_skiptrace_phone (
                        case_id, number, type, carrier,
                        last_reported, score, tested, reachable, dnc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case_id,
                        number,
                        phone_type,
                        carrier,
                        last_reported,
                        score,
                        tested,
                        reachable,
                        dnc,
                    ),
                )

            for em in emails:
                if isinstance(em, dict):
                    email_addr = em.get("email")
                    tested_val = em.get("tested")
                    tested_int = int(tested_val) if isinstance(tested_val, bool) else None
                else:
                    email_addr = str(em)
                    tested_int = None

                if not email_addr:
                    continue

                conn.exec_driver_sql(
                    """
                    INSERT INTO case_skiptrace_email (
                        case_id, email, tested
                    )
                    VALUES (?, ?, ?)
                    """,
                    (case_id, email_addr, tested_int),
                )

    except Exception as exc:
        logger.warning("Failed to save skip trace rows for case %s: %s", case_id, exc)


def load_property_for_case(case_id: int) -> Optional[dict]:
    """
    Load raw property lookup JSON for a case, if it exists.
    """
    try:
        with engine.connect() as conn:
            row = conn.exec_driver_sql(
                "SELECT raw_json FROM case_property WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except Exception:
                return None
    except Exception as exc:
        logger.warning("Failed to load property lookup for case %s: %s", case_id, exc)
    return None


def load_skiptrace_for_case(case_id: int) -> Optional[dict]:
    """
    Load normalized skip-trace data from case_skiptrace + phone/email tables
    and convert it back into the 'skip_trace' dict structure the template expects.
    """
    try:
        with engine.connect() as conn:
            base = conn.exec_driver_sql(
                """
                SELECT
                    owner_name,
                    prop_street, prop_city, prop_state, prop_zip
                FROM case_skiptrace
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()

            phones_rows = conn.exec_driver_sql(
                """
                SELECT
                    number, type, carrier, last_reported,
                    score, tested, reachable, dnc
                FROM case_skiptrace_phone
                WHERE case_id = ?
                ORDER BY
                    CASE WHEN score IS NULL THEN 1 ELSE 0 END,
                    score DESC
                """,
                (case_id,),
            ).fetchall()

            emails_rows = conn.exec_driver_sql(
                """
                SELECT
                    email, tested
                FROM case_skiptrace_email
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchall()
    except Exception as exc:
        logger.warning("Failed to load skip trace for case %s: %s", case_id, exc)
        return None

    if not base:
        return None

    (
        owner_name,
        prop_street, prop_city, prop_state, prop_zip,
    ) = base

    def as_bool(val):
        if val is None:
            return None
        return bool(val)

    phones = []
    for row in phones_rows:
        (
            number, ptype, carrier, last_reported,
            score, tested, reachable, dnc,
        ) = row
        phones.append(
            {
                "number": number,
                "type": ptype,
                "carrier": carrier,
                "last_reported": last_reported,
                "score": score,
                "tested": as_bool(tested),
                "reachable": as_bool(reachable),
                "dnc": as_bool(dnc),
            }
        )

    emails = []
    for row in emails_rows:
        email_addr, tested = row
        emails.append({"email": email_addr, "tested": as_bool(tested)})

    property_address = {
        "street": prop_street,
        "city": prop_city,
        "state": prop_state,
        "postalCode": prop_zip,
    }

    person = {
        "full_name": owner_name,
        "phones": phones,
        "emails": emails,
    }

    return {
        "results": [
            {
                "propertyAddress": property_address,
                "persons": [person],
            }
        ]
    }
