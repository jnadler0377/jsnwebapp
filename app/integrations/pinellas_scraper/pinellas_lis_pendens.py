"""
Scrape Pinellas official records for LIS PENDENS documents within a date range.

Usage (from project root):
    python -m app.integrations.pinellas_scraper.pinellas_lis_pendens --since-days 7 --out data/pinellas_lis_pendens.csv

This uses Playwright to:
    1. Open the Doc Type search page
    2. Accept the disclaimer (if present)
    3. Select Doc Type: "LIS PENDENS"
    4. Set record date range [today - since_days, today]
    5. Click Search
    6. Scrape results grid and write CSV
"""

import asyncio
import csv
import datetime as dt
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from playwright.async_api import async_playwright

# Correct .gov URL
BASE_URL = "https://officialrecords.mypinellasclerk.gov/search/SearchTypeDocType"


def _today() -> dt.date:
    return dt.date.today()


def _date_range_from_since_days(since_days: int) -> tuple[dt.date, dt.date]:
    since_days = max(0, int(since_days))
    today = _today()
    start = today - dt.timedelta(days=since_days)
    return start, today


def _fmt_date(d: dt.date) -> str:
    # Site uses mm/dd/yyyy
    return d.strftime("%m/%d/%Y")


async def _accept_disclaimer(page):
    """
    Accept the disclaimer if we're on the Disclaimer page.
    URL example:
    https://officialrecords.mypinellasclerk.gov/search/Disclaimer?st=/search/SearchTypeDocType
    """
    if "Disclaimer" not in page.url:
        return

    # Try specific button first
    for sel in [
        "#btnAccept",
        "input[type='submit'][value='Accept']",
        "text=Accept",
    ]:
        try:
            btn = await page.wait_for_selector(sel, timeout=3000)
            await btn.click()
            await page.wait_for_timeout(1500)
            return
        except Exception:
            continue

    # If we couldn't click anything, just continue; maybe it's already accepted.


async def _set_doc_type(page):
    """
    Set the Doc Type dropdown to 'LIS PENDENS'.

    The display input is:
        <input class="t-input" id="DocTypesDisplay-input" ... value="All">
    """
    try:
        doc_input = await page.wait_for_selector("#DocTypesDisplay-input", timeout=8000)
    except Exception:
        print("ERROR: Could not find doc type input #DocTypesDisplay-input", file=sys.stderr)
        return

    # Click, type, and select from list if it shows up.
    await doc_input.click()
    await doc_input.fill("")          # clear "All"
    await doc_input.type("LIS PENDENS")

    # Try to click the dropdown option that says "LIS PENDENS"
    # The popup list is usually a <ul> with <li> entries.
    try:
        option = await page.wait_for_selector("li:has-text('LIS PENDENS')", timeout=5000)
        await option.click()
    except Exception:
        # If we can't find the list item, we'll just trust the text in the input.
        print("WARNING: Could not click LIS PENDENS option; relying on typed text.", file=sys.stderr)


async def _set_date_range(page, since_days: int):
    """
    Leave 'Date Range' dropdown on 'Specify Date Range...' and
    directly type into 'From Record Date' and 'To Record Date' fields.

    We don't know their exact IDs, so we use XPath based on label text:
        'From Record Date'
        'To Record Date'
    """
    start, end = _date_range_from_since_days(since_days)
    start_str = _fmt_date(start)
    end_str = _fmt_date(end)

    # From Record Date input
    from_input = None
    for xpath in [
        "xpath=//td[contains(normalize-space(),'From Record Date')]/following::input[1]",
        "xpath=//label[contains(normalize-space(),'From Record Date')]/following::input[1]",
    ]:
        try:
            from_input = await page.wait_for_selector(xpath, timeout=2000)
            break
        except Exception:
            continue

    if from_input is None:
        print("WARNING: Could not find 'From Record Date' input. Adjust XPath.", file=sys.stderr)
    else:
        await from_input.click()
        await from_input.fill("")
        await from_input.type(start_str)

    # To Record Date input
    to_input = None
    for xpath in [
        "xpath=//td[contains(normalize-space(),'To Record Date')]/following::input[1]",
        "xpath=//label[contains(normalize-space(),'To Record Date')]/following::input[1]",
    ]:
        try:
            to_input = await page.wait_for_selector(xpath, timeout=2000)
            break
        except Exception:
            continue

    if to_input is None:
        print("WARNING: Could not find 'To Record Date' input. Adjust XPath.", file=sys.stderr)
    else:
        await to_input.click()
        await to_input.fill("")
        await to_input.type(end_str)


async def _click_search(page):
    """
    Click the Search button:
        <input type="submit" id="btnSearch" value="Search" class="t-button">
    """
    try:
        btn = await page.wait_for_selector("#btnSearch", timeout=5000)
    except Exception:
        print("ERROR: Could not find Search button #btnSearch", file=sys.stderr)
        return

    await btn.click()
    # Let the results load; Acclaim is pretty chatty
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)


async def _run_search(page, since_days: int):
    """
    Full search sequence:
      - set doc type
      - set date range
      - click Search
    """
    await _set_doc_type(page)
    await _set_date_range(page, since_days)
    await _click_search(page)


async def _scrape_grid(page) -> List[Dict[str, Any]]:
    """
    Scrape the search results grid into a list of dicts.

    You may want to adjust the column mapping once you see real data.
    """
    rows_data: List[Dict[str, Any]] = []

    table_selectors = [
        "table[role='grid']",
        "table.t-grid-table",
        "table",  # last resort
    ]

    table = None
    for sel in table_selectors:
        try:
            table = await page.wait_for_selector(sel, timeout=5000)
            if table:
                break
        except Exception:
            continue

    if not table:
        print("ERROR: Could not find results table. Adjust selectors.", file=sys.stderr)
        return rows_data

    row_elements = await table.query_selector_all("tbody tr")
    if not row_elements:
        print("No result rows found.", file=sys.stderr)
        return rows_data

    for row in row_elements:
        cells = await row.query_selector_all("td")
        texts: List[str] = []
        for c in cells:
            txt = (await c.inner_text()).strip()
            texts.append(txt)

        if not any(texts):
            continue

        # Basic mapping – adjust indices once you inspect real results
        row_dict: Dict[str, Any] = {
            "county": "Pinellas",
            "instrument_number": texts[0] if len(texts) > 0 else "",
            "record_date": texts[1] if len(texts) > 1 else "",
            "doc_type": texts[2] if len(texts) > 2 else "",
            "party_1": texts[3] if len(texts) > 3 else "",
            "party_2": texts[4] if len(texts) > 4 else "",
            "case_number": texts[5] if len(texts) > 5 else "",
            "raw": " | ".join(texts),
        }
        rows_data.append(row_dict)

    return rows_data


async def scrape_pinellas_lis_pendens(since_days: int, out_path: Path):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            # Go to doc-type search; if disclaimer appears, accept it
            await page.goto(BASE_URL, wait_until="domcontentloaded")
            await _accept_disclaimer(page)
            # In some cases the accept redirects; if not, manually ensure we're on the search page:
            if "SearchTypeDocType" not in page.url:
                await page.goto(BASE_URL, wait_until="domcontentloaded")

            await _run_search(page, since_days)
            rows = await _scrape_grid(page)
        finally:
            await browser.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "county",
        "instrument_number",
        "record_date",
        "doc_type",
        "party_1",
        "party_2",
        "case_number",
        "raw",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} Pinellas LIS PENDENS rows to {out_path}")


def main(argv: Optional[list[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Pinellas LIS PENDENS records.")
    parser.add_argument("--since-days", type=int, default=7, help="How many days back to search.")
    parser.add_argument(
        "--out",
        type=str,
        default="data/pinellas_lis_pendens.csv",
        help="Path to output CSV.",
    )

    args = parser.parse_args(argv)
    out_path = Path(args.out)

    asyncio.run(scrape_pinellas_lis_pendens(args.since_days, out_path))


if __name__ == "__main__":
    main()
