#!/usr/bin/env python3
"""
Wikipedia GPU Metadata Collector
- Crawls Wikipedia Category pages for GPU-related articles (via MediaWiki API)
- Scrapes each page for metadata (primarily from the infobox and spec tables)
- Writes results to JSON with source URLs for traceability

Outputs:
- gpus.json (structured list of GPU entries)

Install:
  pip install requests beautifulsoup4 python-dateutil

Run:
  python wiki_gpu_scrape.py --limit 200 --out gpus.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser


WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_BASE = "https://en.wikipedia.org/wiki/"

# Categories that often contain GPU product pages. Some pages will be noise; we filter later.
GPU_CATEGORIES = [
    "Category:Graphics_processing_units",
    "Category:Video_cards",
    "Category:Nvidia_graphics_processing_units",
    "Category:AMD_graphics_processing_units",
    "Category:Intel_graphics_processing_units",
]


@dataclass
class ScrapeConfig:
    limit: int
    sleep_s: float
    user_agent: str


def mw_api_get(params: Dict[str, Any], cfg: ScrapeConfig) -> Dict[str, Any]:
    headers = {"User-Agent": cfg.user_agent}
    r = requests.get(WIKI_API, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    time.sleep(cfg.sleep_s)
    return r.json()


def get_category_members(category: str, cfg: ScrapeConfig, cap: int) -> List[Dict[str, Any]]:
    """
    Returns list of pages in a category using MediaWiki API categorymembers.
    """
    members: List[Dict[str, Any]] = []
    cmcontinue: Optional[str] = None

    while len(members) < cap:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "max",
            "cmtype": "page",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = mw_api_get(params, cfg)
        batch = data.get("query", {}).get("categorymembers", [])
        members.extend(batch)

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return members[:cap]


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def clean_text(s: str) -> str:
    s = re.sub(r"\[\d+\]", "", s)  # remove citation markers like [1]
    s = s.replace("\xa0", " ")
    return normalize_ws(s)


def guess_manufacturer(text: str) -> Optional[str]:
    t = text.lower()
    # Conservative detection
    if "nvidia" in t or "geforce" in t or "quadro" in t or "tesla" in t:
        return "NVIDIA"
    if "amd" in t or "radeon" in t or "ati" in t:
        return "AMD"
    if "intel" in t or "arc " in t or "xe " in t:
        return "Intel"
    return None


def parse_wikipedia_html(title: str, cfg: ScrapeConfig) -> Tuple[BeautifulSoup, str]:
    """
    Fetches page HTML via MediaWiki API 'parse' to get reliable HTML content.
    Returns soup and canonical URL.
    """
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text|displaytitle",
        "redirects": 1,
    }
    data = mw_api_get(params, cfg)
    parse = data.get("parse", {})
    html = parse.get("text", {}).get("*", "")
    soup = BeautifulSoup(html, "html.parser")
    # canonical URL
    url = WIKI_BASE + title.replace(" ", "_")
    return soup, url


def extract_infobox_kv(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Extracts key-value pairs from the first infobox on the page.
    Many GPU pages have an infobox with th/td rows.
    """
    infobox = soup.find("table", class_=re.compile(r"\binfobox\b"))
    if not infobox:
        return {}

    kv: Dict[str, str] = {}
    for row in infobox.find_all("tr"):
        th = row.find("th")
        td = row.find("td")
        if not th or not td:
            continue
        key = clean_text(th.get_text(" ", strip=True)).strip(":").lower()
        val = clean_text(td.get_text(" ", strip=True))
        if key and val:
            # keep last occurrence
            kv[key] = val
    return kv


def first_match_in_kv(kv: Dict[str, str], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in kv and kv[k]:
            return kv[k]
    return None


def extract_numeric_with_unit(text: str, unit_regex: str) -> Optional[str]:
    """
    Returns a normalized snippet like '300 W', '256-bit', '16 GB', '19 Gbps', etc.
    """
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(" + unit_regex + r")", text, re.IGNORECASE)
    if not m:
        return None
    num = m.group(1)
    unit = m.group(2)
    return normalize_ws(f"{num} {unit}")


def parse_release_date(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    # Wikipedia often has "September 20, 2022" or "2022" etc.
    try:
        dt = dateparser.parse(raw, fuzzy=True, default=datetime(1900, 1, 1))
        # If only year is present, dateutil may default month/day; detect that by checking if raw is just year-ish.
        if re.fullmatch(r"\d{4}", raw.strip()):
            return raw.strip()
        return dt.date().isoformat()
    except Exception:
        return raw.strip()


def parse_tdp(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Nested JSON only for TDP description, per your request.
    Produces:
      "tdp": {"value": "300 W", "notes": "..."}  (notes optional)
    """
    if not raw:
        return None
    raw_clean = clean_text(raw)

    # Prefer the first wattage found as "value"
    value = extract_numeric_with_unit(raw_clean, r"w|watt|watts")
    if value:
        # normalize W spelling
        value = re.sub(r"\b(watt|watts)\b", "W", value, flags=re.IGNORECASE)
        value = re.sub(r"\bw\b", "W", value, flags=re.IGNORECASE)

    # Notes: keep the remainder (if any) without duplicating the value
    notes = raw_clean
    if value:
        # remove the first occurrence roughly
        notes = re.sub(re.escape(value), "", notes, count=1).strip(" ,;:-")
    if not notes or notes == raw_clean and value is None:
        notes = None

    out: Dict[str, Any] = {"raw": raw_clean}
    if value:
        out["value"] = value
    if notes:
        out["notes"] = notes
    return out


def extract_benchmarks(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    """
    Attempts to find benchmark-like data in tables or sections.
    This is best-effort because Wikipedia benchmark info is inconsistent.
    We look for:
      - tables whose caption or headers mention 'benchmark', '3DMark', 'PassMark', 'GFLOPS', 'FPS'
    Returns a small structured dict with snippets.
    """
    keywords = ["benchmark", "3dmark", "passmark", "gflops", "fps", "performance", "compute"]
    tables = soup.find_all("table", class_=re.compile(r"wikitable"))
    hits: List[Dict[str, Any]] = []

    def table_text(t) -> str:
        return clean_text(t.get_text(" ", strip=True)).lower()

    for t in tables:
        txt = table_text(t)
        if any(k in txt for k in keywords):
            # extract headers + first 3 data rows
            headers = []
            thead = t.find("tr")
            if thead:
                headers = [clean_text(x.get_text(" ", strip=True)) for x in thead.find_all(["th", "td"])]
            rows = []
            for r in t.find_all("tr")[1:4]:
                cols = [clean_text(x.get_text(" ", strip=True)) for x in r.find_all(["th", "td"])]
                if cols:
                    rows.append(cols)
            if headers or rows:
                hits.append({"headers": headers, "rows_preview": rows})

        if len(hits) >= 2:
            break

    if not hits:
        return None

    return {"tables_preview": hits}


def likely_gpu_page(title: str, infobox_kv: Dict[str, str]) -> bool:
    """
    Filters out unrelated pages from broad categories.
    Heuristics:
      - Has an infobox with fields like 'memory', 'tdp', 'architecture', 'codename', 'launch', etc.
      - Or title includes common GPU line terms.
    """
    t = title.lower()
    title_signals = ["geforce", "radeon", "quadro", "tesla", "arc", "rtx", "gtx", "rx ", "intel", "nvidia", "amd"]
    kv_signals = ["memory", "tdp", "architecture", "codename", "launch", "release", "bus", "core clock", "gpu clock"]

    if any(s in t for s in title_signals):
        return True
    if any(k in infobox_kv for k in kv_signals):
        return True
    return False


def extract_gpu_metadata(title: str, cfg: ScrapeConfig) -> Optional[Dict[str, Any]]:
    soup, url = parse_wikipedia_html(title, cfg)
    kv = extract_infobox_kv(soup)

    if not likely_gpu_page(title, kv):
        return None

    name = clean_text(BeautifulSoup(f"<div>{title}</div>", "html.parser").get_text())

    # Manufacturer candidates
    manufacturer_raw = first_match_in_kv(kv, ["manufacturer", "vendor", "developed by", "developer", "produced by"])
    manufacturer = guess_manufacturer(manufacturer_raw or title or "")
    if not manufacturer and manufacturer_raw:
        manufacturer = guess_manufacturer(manufacturer_raw)

    architecture = first_match_in_kv(kv, ["architecture", "microarchitecture", "codename", "core"])
    release_raw = first_match_in_kv(kv, ["release date", "launched", "launch", "introduced", "release"])
    release_date = parse_release_date(release_raw)

    vram = first_match_in_kv(kv, ["memory", "memory size", "vram", "framebuffer", "memory capacity"])
    memory_type = first_match_in_kv(kv, ["memory type", "memory technology"])
    bus_width = first_match_in_kv(kv, ["bus interface", "memory bus", "bus width"])
    core_clock = first_match_in_kv(kv, ["core clock", "gpu clock", "clock"])
    tdp_raw = first_match_in_kv(kv, ["tdp", "power", "thermal design power"])

    # Light normalization for some fields (best-effort, do not overfit)
    if bus_width:
        # try to extract "xxx-bit"
        bw = re.search(r"(\d{2,4})\s*-\s*bit|(\d{2,4})\s*bit", bus_width, re.IGNORECASE)
        if bw:
            bus_width = f"{bw.group(1) or bw.group(2)}-bit"

    if vram:
        # prefer a GB/MB snippet if present
        v = extract_numeric_with_unit(vram, r"gb|gib|mb|mib")
        if v:
            vram = re.sub(r"\b(gib|gb)\b", "GB", v, flags=re.IGNORECASE)
            vram = re.sub(r"\b(mib|mb)\b", "MB", vram, flags=re.IGNORECASE)

    benchmarks = extract_benchmarks(soup)

    entry: Dict[str, Any] = {
        "gpu_name": name,
        "manufacturer": manufacturer,
        "architecture": architecture,
        "specs": {
            "vram": vram,
            "bus_width": bus_width,
            "core_clock": core_clock,
            "memory_type": memory_type,
            # Nested JSON only for TDP description
            "tdp": parse_tdp(tdp_raw),
        },
        "release_date": release_date,
        "benchmarks": benchmarks,
        "source_url": url,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="Max number of GPU pages to collect")
    ap.add_argument("--out", type=str, default="gpus.json", help="Output JSON file")
    ap.add_argument("--sleep", type=float, default=0.25, help="Sleep between requests (seconds)")
    ap.add_argument(
        "--user-agent",
        type=str,
        default="WikiGPUMetadataCollector/1.0 (contact: example@example.com)",
        help="Custom User-Agent (recommended by Wikipedia).",
    )
    args = ap.parse_args()

    cfg = ScrapeConfig(limit=args.limit, sleep_s=args.sleep, user_agent=args.user_agent)

    # 1) Get candidate pages from multiple categories
    seen_titles = set()
    candidates: List[str] = []

    per_cat_cap = max(50, cfg.limit)  # fetch enough; we'll dedupe/filter later
    for cat in GPU_CATEGORIES:
        members = get_category_members(cat, cfg, cap=per_cat_cap)
        for m in members:
            title = m.get("title")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            candidates.append(title)

    # 2) Scrape each page until we reach limit
    results: List[Dict[str, Any]] = []
    for title in candidates:
        if len(results) >= cfg.limit:
            break
        try:
            entry = extract_gpu_metadata(title, cfg)
            if entry:
                results.append(entry)
                print(f"[OK] {entry['gpu_name']}")
        except requests.HTTPError as e:
            print(f"[HTTP] {title}: {e}")
        except Exception as e:
            print(f"[ERR] {title}: {e}")

    # 3) Write JSON
    payload = {
        "collection": "wikipedia_gpu_metadata",
        "source": "Wikipedia (MediaWiki API + HTML parsing)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(results),
        "items": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(results)} items to {args.out}")


if __name__ == "__main__":
    main()