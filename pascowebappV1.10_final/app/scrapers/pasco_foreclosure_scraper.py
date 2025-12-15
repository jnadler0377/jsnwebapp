# Pasco Clerk Subscriber Foreclosure Scraper + Civitek downloader (CLEAN)
# See chat for usage notes.
from __future__ import annotations
import os, sys, time, re, random, argparse, logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urljoin

from dotenv import load_dotenv
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout, Page

PORTAL_LOGIN_URL = "https://app.pascoclerk.com/appdot-private-subsc-form-courts-nc-foreclosures.asp"
RESULTS_URL      = "https://app.pascoclerk.com/appdot-private-subsc-results-courts-nc-fc-lispend30.asp"
CIVITEK_ROOT     = "https://www.civitekflorida.com/ocrs/county/51/"

RESULTS_TABLE = "div.results_courtsdiv > table"
ROW_SELECTOR  = f"{RESULTS_TABLE} tr:has(td)"
CELL_DATE     = "td:nth-child(1)"
CELL_CASE     = "td:nth-child(2)"
CELL_INFO     = "td:nth-child(3) a"

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)
LOG_PATH = os.path.join(DEBUG_DIR, "scraper.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("pasco-foreclosure")

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def parse_mdy(text: str):
    try: return datetime.strptime(text.strip(), "%m/%d/%Y")
    except Exception: return None
def safe_text(el): 
    try: return el.inner_text().strip()
    except Exception: return ""
def human_delay(max_sec: int, *, min_sec: float = 0.3):
    if max_sec <= 0: return
    time.sleep(random.uniform(min_sec, max(max_sec, min_sec)))
def human_type(page, selector: str, text: str, humanize_max: int):
    loc = page.locator(selector); loc.click()
    for ch in text: loc.type(ch); time.sleep(random.uniform(0.05, 0.25))
    human_delay(min(1, humanize_max), min_sec=0.2)

def click_lis_pendens_filed(page_or_frame, humanize_max: int) -> bool:
    import re as _re
    human_delay(min(3, humanize_max))
    strategies = [
        page_or_frame.get_by_role("button", name=_re.compile(r"lis\s*pendens\s*filed", _re.I)),
        page_or_frame.locator("button:has-text('Lis Pendens Filed')"),
        page_or_frame.locator("input[type='submit'][value*='Lis' i][value*='Pendens' i]"),
        page_or_frame.get_by_role("button", name=_re.compile(r"(submit|search|apply)", _re.I)).filter(has_text=_re.compile(r"lis\s*pendens", _re.I)),
    ]
    for loc in strategies:
        try:
            if loc.count() > 0:
                loc = loc.first; loc.scroll_into_view_if_needed()
                with page_or_frame.expect_navigation(wait_until="networkidle", timeout=15000): loc.click()
                log.info("Clicked 'Lis Pendens Filed'"); return True
        except Exception: pass
    for target in ["ctl00$Main$btnLisPendens", "Main$btnLisPendens", "btnLisPendens"]:
        try:
            page_or_frame.evaluate(f"window.__doPostBack && __doPostBack('{target}','')"); page_or_frame.wait_for_load_state("networkidle"); return True
        except Exception: pass
    try: page_or_frame.evaluate("document.forms[0] && document.forms[0].submit()"); page_or_frame.wait_for_load_state("networkidle"); return True
    except Exception: pass
    return False

def collect_rows_and_links(page, since_days: int):
    items = []; rows = page.query_selector_all(ROW_SELECTOR) or []; base = page.url
    cutoff = None
    if since_days and since_days > 0: cutoff = datetime.now() - timedelta(days=since_days)
    for r in rows:
        filed = safe_text(r.query_selector(CELL_DATE)); case = safe_text(r.query_selector(CELL_CASE))
        a = r.query_selector(CELL_INFO); href = a.get_attribute("href") if a else None
        if not (case and href): continue
        if cutoff is not None:
            d = parse_mdy(filed); 
            if d and d < cutoff: continue
        items.append((filed, case, urljoin(base, href)))
    return items

def _norm(s: str) -> str: return re.sub(r"\s+", " ", (s or "").strip())

def _value_next_to_label(page, label_variants: List[str]) -> str:
    for lbl in label_variants:
        sel = f"xpath=//*[self::td or self::th][normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'))= '{lbl.lower()}']/following-sibling::*[1]"
        loc = page.locator(sel); 
        if loc.count() > 0: return _norm(loc.first.inner_text())
        sel = f"xpath=//*[self::td or self::th][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{lbl.lower()}')]/following-sibling::*[1]"
        loc = page.locator(sel); 
        if loc.count() > 0: return _norm(loc.first.inner_text())
        loc = page.locator(f"td:has-text('{lbl}') + td, th:has-text('{lbl}') + td")
        if loc.count() > 0: return _norm(loc.first.inner_text())
    return ""

def _value_inline_colon(page, label_variants: List[str]) -> str:
    for lbl in label_variants:
        loc = page.locator(f"xpath=//*[self::td or self::th][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), '{lbl.lower()}')]")
        if loc.count() > 0:
            text = _norm(loc.first.inner_text())
            if ':' in text: return _norm(text.split(':', 1)[1])
    return ""

def _collect_all_defendants(page) -> List[str]:
    found: List[str] = []
    rows = page.locator("xpath=//tr[./*[(self::td or self::th) and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'defendant')]]")
    for i in range(rows.count()):
        row = rows.nth(i); val = ""
        cell = row.locator("xpath=./*[(self::td or self::th) and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'defendant')]/following-sibling::*[1]")
        if cell.count() > 0:
            val = _norm(cell.first.inner_text())
        else:
            txt = _norm(row.inner_text())
            if ':' in txt: val = _norm(txt.split(':', 1)[1])
        if val: found.append(val)
    headings = page.locator("xpath=//*[self::td or self::th or self::h1 or self::h2 or self::h3][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'defendant')]")
    for i in range(headings.count()):
        h = headings.nth(i)
        lis = h.locator("xpath=following::li[position()<=5]")
        for j in range(min(5, lis.count())):
            txt = _norm(lis.nth(j).inner_text())
            if txt and txt not in found: found.append(txt)
    body_txt = page.inner_text("body")
    for m in re.finditer(r"Defendant[s]?:\s*(.+)", body_txt, flags=re.I):
        frag = _norm(m.group(1).splitlines()[0])
        if frag and frag not in found: found.append(frag)
    seen = set(); unique = []
    for d in found:
        if d not in seen: unique.append(d); seen.add(d)
    return unique

def extract_fields_from_detail(page, fallback_case_no: str, fallback_filed_date: str) -> Dict[str, Any]:
    CASE_NAME_LBL = ["Case Name", "Style", "Style of Case", "Style/Case Name", "Style of the Case"]
    CASE_NO_LBL   = ["Case #", "Case Number", "Case No.", "Case Num", "Case No"]
    FILED_LBL     = ["Filing Date", "Filed", "Date Filed", "Filed Date"]
    case_name = _value_next_to_label(page, CASE_NAME_LBL) or _value_inline_colon(page, CASE_NAME_LBL)
    case_no   = _value_next_to_label(page, CASE_NO_LBL)   or _value_inline_colon(page, CASE_NO_LBL)   or fallback_case_no
    filed_dt  = _value_next_to_label(page, FILED_LBL)     or _value_inline_colon(page, FILED_LBL)     or fallback_filed_date
    defendants = _collect_all_defendants(page)
    return {"Case Name": case_name, "Case #": case_no, "Filing Date": filed_dt, "Defendants": defendants}

def parse_case_components(case_str: str) -> Optional[Tuple[str, str, str]]:
    s = (case_str or "").strip()
    m = re.search(r"(?P<year>20\d{2}).{0,3}(?P<ct>[A-Z]{2,3}).{0,3}(?P<seq>\d{5,7})", s, flags=re.I)
    if m: return (m.group("year"), m.group("ct").upper(), m.group("seq"))
    try:
        year = s[3:7]; ct = s[8:10].upper(); seq = re.sub(r"\D", "", s[11:17])
        if len(year)==4 and len(ct)>=2 and len(seq)>=5: return (year, ct, seq)
    except Exception: pass
    return None

def civitek_open_portal(page: Page, humanize_max: int):
    page.goto(CIVITEK_ROOT, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_load_state("networkidle"); human_delay(min(2, humanize_max))
    try: page.get_by_role("link", name=re.compile(r"public", re.I)).click(timeout=8000)
    except Exception: page.click("a:has-text('Public')", timeout=8000)
    page.wait_for_load_state("networkidle"); human_delay(min(1, humanize_max))
    try: page.get_by_role("link", name=re.compile(r"agree", re.I)).click(timeout=8000)
    except Exception: page.click("a:has-text('I Agree')", timeout=8000)
    page.wait_for_load_state("networkidle"); human_delay(min(1, humanize_max))
    try: page.get_by_role("tab", name=re.compile(r"case\s*search", re.I)).click(timeout=8000)
    except Exception: page.click("text=Case Search", timeout=8000)
    page.wait_for_load_state("networkidle"); human_delay(min(1, humanize_max))
    try:
        frames = page.frames; clicked = False
        for fr in frames:
            try:
                box = fr.locator("input[type='checkbox'], span[role='checkbox']")
                if box.count() > 0: box.first.click(timeout=5000); clicked = True; break
            except Exception: continue
        if not clicked: page.locator("input[type='checkbox'], span[role='checkbox']").first.click(timeout=5000)
    except Exception:
        log.warning("Cloudflare checkbox not auto-clickable. Waiting 30s for manual completion…"); time.sleep(30)

def civitek_search_and_download(page: Page, context, year: str, court_type: str, seq: str, out_root: str, humanize_max: int):
    human_delay(min(1, humanize_max))
    try: page.locator("input[name='caseyear'], input#caseyear").fill(year)
    except Exception: page.get_by_label(re.compile(r"year", re.I)).fill(year)
    try:
        sel = page.locator("select[name='casetype'], select#casetype"); sel.select_option(court_type.upper())
    except Exception: page.select_option("select", label=court_type.upper())
    try: page.locator("input[name='caseseq'], input#caseseq").fill(seq)
    except Exception: page.get_by_label(re.compile(r"sequence", re.I)).fill(seq)
    human_delay(min(1, humanize_max))
    try: page.get_by_role("button", name=re.compile(r"search", re.I)).click(timeout=10000)
    except Exception: page.click("button:has-text('SEARCH'), input[type='submit'][value*='SEARCH' i]")
    page.wait_for_load_state("networkidle"); human_delay(min(2, humanize_max))
    try:
        firstRow = page.locator("table tr:has(td) a").first
        if firstRow: firstRow.click(timeout=8000); page.wait_for_load_state("networkidle"); human_delay(min(2, humanize_max))
    except Exception: pass
    targets = [re.compile(r"verified\s+complaint\s+to\s+foreclosure\s+mortgage", re.I),
               re.compile(r"value\s+calculation\s+for\s+real\s+property", re.I)]
    out_dir = os.path.join(out_root, seq); ensure_dir(out_dir)
    links = page.locator("a"); count = links.count()
    for i in range(count):
        a = links.nth(i)
        try: text = (a.inner_text() or "").strip()
        except Exception: continue
        for pat in targets:
            if pat.search(text or ""):
                try:
                    with page.expect_download() as dl_info: a.click()
                    download = dl_info.value
                    suggested = download.suggested_filename
                    dest = os.path.join(out_dir, suggested or f"{seq}_{i}.pdf")
                    download.save_as(dest); log.info(f"Downloaded → {dest}")
                except Exception as e: log.warning(f"Failed downloading '{text}': {e}")
                break

def phase1_scrape_and_save(args) -> str:
    load_dotenv()
    user = os.getenv("PASCO_USER", "")
    pwd = os.getenv("PASCO_PASS", "")
    out_root = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "output"))
    ensure_dir(out_root)

    if not user or not pwd:
        raise RuntimeError("Missing PASCO_USER/PASCO_PASS in .env")

    # Allow --out override from command line
    results: List[Dict[str, Any]] = []
    csv_path = (args.out.strip() or os.path.join(out_root, "pasco_foreclosures.csv"))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        context = browser.new_context(accept_downloads=True)
        if args.trace:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()

        try:
            log.info("Navigating to Pasco subscriber login…")
            page.goto(PORTAL_LOGIN_URL, wait_until="domcontentloaded", timeout=60000)
            human_type(page, "input[type='text'], input[type='email']", user, args.humanize_max)
            human_delay(min(1, args.humanize_max))
            human_type(page, "input[type='password']", pwd, args.humanize_max)
            human_delay(min(1, args.humanize_max))
            try:
                page.click("button[type='submit'], input[type='submit']", timeout=15000)
            except PlaywrightTimeout:
                page.evaluate("document.forms[0] && document.forms[0].submit()")
            page.wait_for_load_state("networkidle", timeout=20000)

            clicked = click_lis_pendens_filed(page, args.humanize_max)
            if not clicked:
                for fr in page.frames:
                    try:
                        if click_lis_pendens_filed(fr, args.humanize_max):
                            clicked = True
                            break
                    except Exception:
                        pass
            if not clicked:
                log.warning("Could not click 'Lis Pendens Filed' — fallback to results URL")
                human_delay(min(2, args.humanize_max))
                page.goto(RESULTS_URL, wait_until="domcontentloaded", timeout=45000)
                page.wait_for_load_state("networkidle")

            items = collect_rows_and_links(page, since_days=args.since_days)
            log.info(f"Harvesting {len(items)} info links (since_days={args.since_days})")

            for filed, case_no, info_url in items:
                try:
                    human_delay(args.humanize_max)
                    page.goto(info_url, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_load_state("networkidle")
                    human_delay(min(2, args.humanize_max))
                    fields = extract_fields_from_detail(page, fallback_case_no=case_no, fallback_filed_date=filed)
                    results.append(fields)
                    if args.max_records and len(results) >= args.max_records:
                        break
                except Exception:
                    log.exception(f"Failed on {info_url}")
                    dump_artifacts(page, context, screenshot=args.screenshot, save_html=args.save_html)
                    continue

            max_defs = 0
            for r in results:
                max_defs = max(max_defs, len(r.get("Defendants", [])))

            columns = ["Case Name", "Case #", "Filing Date"] + [f"Defendant {i}" for i in range(1, max_defs + 1)]
            rows_out = []
            for r in results:
                row = {
                    "Case Name": r.get("Case Name", ""),
                    "Case #": r.get("Case #", ""),
                    "Filing Date": r.get("Filing Date", ""),
                }
                defs = r.get("Defendants", [])
                for i in range(max_defs):
                    row[f"Defendant {i+1}"] = defs[i] if i < len(defs) else ""
                rows_out.append(row)

            pd.DataFrame(rows_out, columns=columns).to_csv(csv_path, index=False, encoding="utf-8-sig")
            log.info("Saved %d rows -> %s", len(rows_out), csv_path)

        except Exception:
            log.exception("Fatal error during Phase 1")
            dump_artifacts(page, context, screenshot=args.screenshot, save_html=args.save_html, trace=args.trace)
            raise

        finally:
            try:
                if args.trace:
                    dump_artifacts(page, context, trace=True)
                context.close()
                browser.close()
            except Exception:
                pass

    return csv_path


def phase2_civitek_download(args, csv_path: str):
    out_root = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "output"))
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    cases = df["Case #"].tolist()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        context = browser.new_context(accept_downloads=True)
        if args.trace: context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        civitek_open_portal(page, args.humanize_max)
        for case_str in cases:
            parts = parse_case_components(case_str)
            if not parts: log.warning(f"Could not parse Case # '{case_str}', skipping."); continue
            year, ct, seq = parts; log.info(f"Civitek search → year={year}, type={ct}, seq={seq}")
            try:
                civitek_search_and_download(page, context, year, ct, seq, out_root, args.humanize_max)
                human_delay(min(2, args.humanize_max))
                try: page.get_by_role("tab", name=re.compile(r"case\s*search", re.I)).click(timeout=8000)
                except Exception: page.click("text=Case Search", timeout=8000)
            except Exception:
                log.exception(f"Failed Civitek processing for {case_str}"); continue
        try:
            if args.trace: dump_artifacts(page, context, trace=True)
            context.close(); browser.close()
        except Exception: pass

def parse_args():
    p = argparse.ArgumentParser(description="Pasco foreclosure scraper + Civitek downloader")
    p.add_argument("--headless", action="store_true", help="Run headless (for scheduler)")
    p.add_argument("--trace", action="store_true", help="Record Playwright trace to debug/trace.zip")
    p.add_argument("--screenshot", action="store_true", help="Save a screenshot on failures")
    p.add_argument("--save-html", action="store_true", help="Save page HTML on failures")
    p.add_argument("--debug", action="store_true", help="Enable Playwright inspector (PWDEBUG=1)")
    p.add_argument("--humanize-max", type=int, default=30, help="Max seconds for random human-like pauses (default 30)")
    p.add_argument("--since-days", type=int, default=0, help="Only include filings within the last N days (0 = no filter)")
    p.add_argument("--max-records", type=int, default=0, help="Stop after N records (0 = no limit)")
    p.add_argument("--civitek", action="store_true", help="Run Phase 2: Civitek search + downloads")

    # NEW: allow main app to specify output CSV path
    p.add_argument("--out", type=str, default="", help="Write harvested CSV to this exact path")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug: os.environ["PWDEBUG"] = "1"
    csv_path = phase1_scrape_and_save(args)
    if args.civitek: phase2_civitek_download(args, csv_path)

if __name__ == "__main__":
    try: main()
    except Exception as e: log.error(f"EXIT WITH ERROR: {e}"); sys.exit(1)
    log.info("DONE")
