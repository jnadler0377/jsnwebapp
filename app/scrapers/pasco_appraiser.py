import os
import re
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

APPRAISER_BASE = "https://search.pascopa.com"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/118 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}


def build_pasco_url(parcel_id: str) -> Optional[str]:
    """Return the Appraiser parcel.aspx URL for a parcel id like 25-26-15-0060-00000-0470."""
    if not parcel_id:
        return None
    parts = parcel_id.split("-")
    if len(parts) != 6:
        return None
    sec, twn, rng, sbb, blk, lot = parts
    return (
        f"{APPRAISER_BASE}/parcel.aspx?sec={sec}&twn={twn}&rng={rng}"
        f"&sbb={sbb}&blk={blk}&lot={lot}&action=Submit"
    )


def _client_kwargs():
    def _env_float(k: str, d: float) -> float:
        try:
            return float(os.getenv(k, d))
        except Exception:
            return float(d)

    verify = os.getenv("PASCO_INSECURE_SSL", "").strip() != "1"
    timeout = httpx.Timeout(
        connect=_env_float("PASCO_TIMEOUT_CONNECT", 45),
        read=_env_float("PASCO_TIMEOUT_READ", 90),
        write=_env_float("PASCO_TIMEOUT_WRITE", 45),
        pool=_env_float("PASCO_TIMEOUT_POOL", 45),
    )
    limits = httpx.Limits(max_connections=5, max_keepalive_connections=0)

    return dict(
        verify=verify,
        http2=False,
        timeout=timeout,
        limits=limits,
        headers=DEFAULT_HEADERS.copy(),
        follow_redirects=True,
    )


def _get_with_retries(
    client: httpx.Client,
    url: str,
    referer: Optional[str] = None,
    max_attempts: Optional[int] = None,
) -> httpx.Response:
    if max_attempts is None:
        try:
            max_attempts = int(os.getenv("PASCO_RETRIES", "5"))
        except Exception:
            max_attempts = 5

    headers = {}
    if referer:
        headers["Referer"] = referer

    last_err: Optional[Exception] = None
    for i in range(max_attempts):
        try:
            return client.get(url, headers=headers)
        except (
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.PoolTimeout,
        ) as e:
            last_err = e
            time.sleep(min(2 ** i, 10) + random.random() * 0.4)

    if last_err:
        raise last_err
    raise RuntimeError("GET failed unexpectedly")


def _looks_like_pdf(resp: httpx.Response, body: bytes) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return ("pdf" in ctype) or body.startswith(b"%PDF-")


def fetch_sales_history_links(parcel_id: str) -> List[Tuple[str, str]]:
    """
    Return (label, href) tuples for Sales History Book/Page links as shown
    on the Appraiser page (newest-first as listed on the site).
    """
    page_url = build_pasco_url(parcel_id)
    if not page_url:
        return []

    with httpx.Client(**_client_kwargs()) as client:
        r = _get_with_retries(client, page_url)
        r.raise_for_status()
        html = r.text

    soup = BeautifulSoup(html, "html.parser")

    # Find the "Sales History" section
    section = None
    for tag in soup.find_all(string=re.compile(r"Sales\s*History", re.I)):
        parent = tag.parent
        for _ in range(4):
            if parent is None:
                break
            if parent.name in ("h2", "h3", "h4", "div", "section"):
                section = parent
                break
            parent = parent.parent
        if section:
            break

    # The table is typically the next <table> after the section header
    table = section.find_next("table") if section else None
    if not table:
        # Fallback: any table that has a header containing "Book/Page"
        for t in soup.find_all("table"):
            heads = [th.get_text(strip=True) for th in t.find_all("th")]
            if any(re.search(r"Book/?Page", h, re.I) for h in heads):
                table = t
                break

    if not table:
        return []

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    bp_idx = next((i for i, h in enumerate(headers) if re.search(r"Book/?Page", h, re.I)), None)
    my_idx = next((i for i, h in enumerate(headers) if re.search(r"(Month|Year|Date)", h, re.I)), None)
    if bp_idx is None:
        return []

    out: List[Tuple[str, str]] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if not cells or len(cells) <= bp_idx:
            continue
        a = cells[bp_idx].find("a", href=True)
        if not a:
            continue
        raw_href = a["href"]
        href = urljoin(page_url, raw_href)  # normalize relative
        left = cells[my_idx].get_text(strip=True) if (my_idx is not None and len(cells) > my_idx) else ""
        label = f"{left} | {a.get_text(strip=True)}".strip(" |")
        out.append((label, href))

    return out


def _same_host(url: str, host_base: str) -> bool:
    try:
        return urlparse(url).netloc.lower().endswith(urlparse(host_base).netloc.lower())
    except Exception:
        return False


def download_two_latest(parcel_id: str, dest_folder: Path) -> List[Path]:
    """
    POLICY:
      • DO NOT attempt to download from Clerk/Official Records (CAPTCHA).
      • Try to download only if the link is still on the Appraiser host.
      • Otherwise, save an HTML file with a manual link.
    """
    items = fetch_sales_history_links(parcel_id)
    if not items:
        return []

    dest_folder.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    with httpx.Client(**_client_kwargs()) as client:
        for (label, href) in items[:2]:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)[:80]

            if _same_host(href, APPRAISER_BASE):
                try:
                    resp = _get_with_retries(client, href, referer=build_pasco_url(parcel_id))
                    body = resp.content
                    if _looks_like_pdf(resp, body):
                        out = dest_folder / f"Appraiser_{safe}.pdf"
                        out.write_bytes(body)
                        saved.append(out)
                        continue
                    else:
                        # Not a PDF — save HTML body so it's at least viewable
                        out = dest_folder / f"Appraiser_{safe}.html"
                        text = body.decode("utf-8", errors="ignore") if body else ""
                        out.write_text(text, encoding="utf-8", errors="ignore")
                        saved.append(out)
                        continue
                except Exception:
                    # Fall back to manual link file
                    note = (
                        f"<html><body>Automatic fetch failed. "
                        f"Open manually: <a href='{href}'>Sales document</a></body></html>"
                    )
                    out = dest_folder / f"Appraiser_{safe}.html"
                    out.write_text(note, encoding="utf-8", errors="ignore")
                    saved.append(out)
                    continue
            else:
                # Off-site (e.g., Clerk with CAPTCHA) → manual link only
                note = (
                    f"<html><body>Manual download required (CAPTCHA). "
                    f"Open: <a href='{href}'>Sales document</a></body></html>"
                )
                out = dest_folder / f"Appraiser_{safe}.html"
                out.write_text(note, encoding="utf-8", errors="ignore")
                saved.append(out)

    return saved
