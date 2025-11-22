import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup


def parse_propwire_html(html: str) -> Dict[str, Any]:
    """
    Parse Propwire HTML to extract key property info.

    This is similar in spirit to the earlier BeautifulSoup approach, but now
    it receives HTML from a real, logged-in browser session (Playwright),
    so we are not fighting 403s anymore.
    """
    out: Dict[str, Any] = {
        "estimated_value": None,
        "sqft": None,
        "lot_size": None,
        "est_equity": None,
        "open_mortgages": None,
        "year_built": None,
        "apn": None,
        "pool": None,
    }

    soup = BeautifulSoup(html, "html.parser")

    # Try JSON blobs first (Next.js-style or ld+json).
    import json as _json

    def harvest_from_obj(obj, bucket: Dict[str, Any]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()

                # Estimated value
                if any(tok in kl for tok in ["estimatedvalue", "estvalue", "avmvalue", "estimated_value"]):
                    if bucket["estimated_value"] is None:
                        bucket["estimated_value"] = v

                # Square feet
                if any(tok in kl for tok in ["sqft", "squarefeet", "livingarea", "buildingareasqft"]):
                    if bucket["sqft"] is None:
                        bucket["sqft"] = v

                # Lot size
                if any(tok in kl for tok in ["lotsize", "lot_sqft", "lotacres", "lot_size"]):
                    if bucket["lot_size"] is None:
                        bucket["lot_size"] = v

                # Equity
                if any(tok in kl for tok in ["estequity", "estimatedequity", "equityamount"]):
                    if bucket["est_equity"] is None:
                        bucket["est_equity"] = v

                # Open mortgages
                if any(tok in kl for tok in ["openmortgages", "open_loans", "openliens", "num_open_mortgages"]):
                    if bucket["open_mortgages"] is None:
                        bucket["open_mortgages"] = v

                # Year built
                if any(tok in kl for tok in ["yearbuilt", "yrbuilt", "constructionyear"]):
                    if bucket["year_built"] is None:
                        bucket["year_built"] = v

                # APN / Parcel ID
                if any(tok in kl for tok in ["apn", "parcelnumber", "parcel_id", "parcelid"]):
                    if bucket["apn"] is None:
                        bucket["apn"] = v

                # Pool
                if "pool" in kl:
                    if bucket["pool"] is None:
                        bucket["pool"] = v

                harvest_from_obj(v, bucket)

        elif isinstance(obj, list):
            for item in obj:
                harvest_from_obj(item, bucket)

    json_candidates = []
    for script in soup.find_all("script"):
        script_type = (script.get("type") or "").lower()
        if script_type not in ("application/json", "application/ld+json") and script.get("id") != "__NEXT_DATA__":
            continue
        raw = (script.string or script.text or "").strip()
        if not raw:
            continue
        try:
            data = _json.loads(raw)
        except Exception:
            continue
        if isinstance(data, dict):
            json_candidates.append(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    json_candidates.append(item)

    for data in json_candidates:
        harvest_from_obj(data, out)

    # Fallback: label-based HTML scanning, in case JSON misses some fields.
    def find_value_by_label(labels):
        for label in labels:
            el = soup.find(string=lambda t: t and label.lower() in t.lower())
            if not el:
                continue
            parent = el.parent
            candidates = [parent.next_sibling, parent.parent.next_sibling]
            for c in candidates:
                if not c:
                    continue
                txt = getattr(c, "get_text", lambda: "")().strip()
                if txt:
                    return txt
        return None

    if out["estimated_value"] is None:
        out["estimated_value"] = find_value_by_label(
            ["Estimated Value", "Est. Property Value"]
        )
    if out["sqft"] is None:
        out["sqft"] = find_value_by_label(["Sq Ft", "Square Feet", "Living Area"])
    if out["lot_size"] is None:
        out["lot_size"] = find_value_by_label(["Lot Size"])
    if out["est_equity"] is None:
        out["est_equity"] = find_value_by_label(["Est. Equity", "Estimated Equity"])
    if out["open_mortgages"] is None:
        out["open_mortgages"] = find_value_by_label(
            ["Open Mortgages", "Open Loan(s)"]
        )
    if out["year_built"] is None:
        out["year_built"] = find_value_by_label(["Year Built"])
    if out["apn"] is None:
        out["apn"] = find_value_by_label(["APN", "Parcel ID"])
    if out["pool"] is None:
        out["pool"] = find_value_by_label(["Pool"])

    return out


async def scrape_propwire_page(url: str, user_data_dir: Path, headless: bool) -> Dict[str, Any]:
    """
    Use Playwright to open Propwire with a persistent Chromium profile.
    - First run: headless=False (interactive) so you can log in once.
    - Later runs: headless=True, reusing the same user_data_dir (cookies/session).
    """
    from playwright.async_api import async_playwright

    user_data_dir = user_data_dir.expanduser().resolve()
    user_data_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=headless,
        )
        page = await context.new_page()
        await page.goto(url, wait_until="networkidle")
        # Give React time to settle if needed
        await page.wait_for_timeout(3000)
        html = await page.content()
        data = parse_propwire_html(html)
        await context.close()
        return data


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Propwire property details via Playwright.")
    parser.add_argument("--url", required=True, help="Propwire property-details URL")
    parser.add_argument(
        "--user-data-dir",
        default=str(Path(__file__).resolve().parent / "pw_profile"),
        help="Directory for persistent Chromium profile (stores login cookies).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run with a visible browser window for login (headless=False).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless browser. (Default is headless unless --interactive is set.)",
    )

    args = parser.parse_args()
    user_data_dir = Path(args.user_data_dir)
    headless = not args.interactive

    # If both flags are set, prefer interactive.
    if args.headless and args.interactive:
        headless = False

    data = asyncio.run(scrape_propwire_page(args.url, user_data_dir, headless=headless))
    print(json.dumps(data))


if __name__ == "__main__":
    main()
