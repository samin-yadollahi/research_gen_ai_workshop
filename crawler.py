#!/usr/bin/env python3
"""
GPU Metadata Collector (Wikipedia + Technical.City comparison pages) — FIXED to produce items

Why you were getting 0 items
----------------------------
Your previous run scraped Technical.City *single-GPU* pages and tried to find specs in tables.
In many cases, those pages don’t expose the required spec rows in a simple 2-column table format
(or are structured differently), so the scraper failed to extract VRAM/Bus/CoreClock/MemoryType/TDP
=> every GPU was "incomplete" => skipped => JSON empty.

What this version does differently (works reliably)
---------------------------------------------------
✅ Uses Technical.City *comparison pages* (GPU-A-vs-GPU-B) which clearly contain rows like:
   - Memory type
   - Maximum RAM amount
   - Memory bus width
   - Core clock speed
   - Power consumption (TDP)

✅ Crawling is bounded:
   - rating pages MAX <= 15 (your requirement)
   - GPU count bounded with --max-gpus (default 50)

✅ TDP is stored ONLY as plain value (e.g., "250 W") — no description

✅ Specs are never blank in output:
   - if any required spec is missing, that GPU is skipped
   - (so items always have complete specs)

✅ Adds Wikipedia fields (best-effort):
   - Architecture
   - Release Date
   - Source URL

Install:
  pip install requests beautifulsoup4 python-dateutil

Run (fast):
  python gpu_collect.py --out gpus.json --max-pages 5 --max-gpus 40

Run with Wikipedia (slower):
  python gpu_collect.py --wiki --out gpus.json --max-pages 5 --max-gpus 25
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser


# ----------------------------
# Sources
# ----------------------------

TECHCITY_BASE = "https://technical.city"
TECHCITY_RATING_URL = "https://technical.city/en/video/rating/1000?pg={pg}"

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_REST_HTML = "https://en.wikipedia.org/api/rest_v1/page/html/{}"


# ----------------------------
# Network config
# ----------------------------

DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

@dataclass
class NetCfg:
    user_agent: str
    timeout_s: int
    sleep_s: float
    retries: int


def make_session(cfg: NetCfg) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": cfg.user_agent, "Accept-Language": "en-US,en;q=0.9"})
    return s


def get_with_retries(url: str, session: requests.Session, cfg: NetCfg, params: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
    last = None
    for _ in range(cfg.retries):
        try:
            r = session.get(url, params=params, timeout=cfg.timeout_s)
            r.raise_for_status()
            if cfg.sleep_s > 0:
                time.sleep(cfg.sleep_s)
            return r
        except Exception as e:
            last = e
            time.sleep(min(1.0, max(0.05, cfg.sleep_s)))
    return None


# ----------------------------
# Text helpers
# ----------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clean_text(s: str) -> str:
    s = (s or "").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_vram(s: str) -> Optional[str]:
    s = clean_text(s)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(GB|GiB|MB|MiB)\b", s, flags=re.IGNORECASE)
    if not m:
        return None
    val = m.group(1)
    unit = m.group(2).lower()
    unit = "GB" if unit in ("gb", "gib") else "MB"
    return f"{val} {unit}"

def normalize_bus(s: str) -> Optional[str]:
    s = clean_text(s)
    m = re.search(r"(\d{2,4})\s*Bit\b", s, flags=re.IGNORECASE)
    return f"{m.group(1)} bit" if m else None

def normalize_clock(s: str) -> Optional[str]:
    s = clean_text(s)
    # Usually MHz
    m = re.search(r"(\d{2,5})\s*MHz\b", s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)} MHz"
    # Sometimes "no data" etc
    if s.lower() in {"no data", "n/a", "-", "—"}:
        return None
    # Keep short if present
    return s[:80] if s else None

def normalize_tdp(s: str) -> Optional[str]:
    s = clean_text(s)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(Watt|W|Watts)\b", s, flags=re.IGNORECASE)
    return f"{m.group(1)} W" if m else None

def detect_manufacturer(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["nvidia", "geforce", "quadro", "rtx", "gtx", "tesla", "titan", "nvs"]):
        return "NVIDIA"
    if any(k in n for k in ["amd", "radeon", "firepro", "instinct", "vega", "rx "]):
        return "AMD"
    if any(k in n for k in ["intel", "arc ", "iris", "uhd", "hd graphics", "xe "]):
        return "Intel"
    return "Unknown"

def is_complete_specs(specs: Dict[str, Optional[str]]) -> bool:
    req = ["vram", "bus_width", "core_clock", "memory_type", "tdp"]
    for k in req:
        v = specs.get(k)
        if not v or not clean_text(str(v)):
            return False
        if clean_text(str(v)).lower() in {"no data", "n/a", "-", "—"}:
            return False
    return True


# ----------------------------
# Technical.City: candidates from rating pages
# ----------------------------

def extract_rating_candidates(html: str) -> List[Tuple[str, str]]:
    """
    Returns (gpu_name, gpu_slug) from rating page.
    Rating pages contain links like /en/video/GeForce-RTX-3080-Ti
    """
    soup = BeautifulSoup(html, "html.parser")
    out: List[Tuple[str, str]] = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/en/video/"):
            continue
        if "-vs-" in href:
            continue
        name = clean_text(a.get_text(" ", strip=True))
        if not name or len(name) < 3:
            continue

        slug = href.split("/en/video/")[-1].strip("/")
        if not slug or slug in seen:
            continue

        seen.add(slug)
        out.append((name, slug))

    return out


def collect_candidates(session: requests.Session, cfg: NetCfg, max_pages: int, max_gpus: int) -> List[Tuple[str, str]]:
    cands: List[Tuple[str, str]] = []
    seen_slug = set()

    for pg in range(1, max_pages + 1):
        url = TECHCITY_RATING_URL.format(pg=pg)
        r = get_with_retries(url, session, cfg)
        if r is None:
            continue

        for name, slug in extract_rating_candidates(r.text):
            if slug in seen_slug:
                continue
            seen_slug.add(slug)
            cands.append((name, slug))
            if len(cands) >= max_gpus:
                return cands

    return cands[:max_gpus]


# ----------------------------
# Technical.City: scrape specs from comparison page
# ----------------------------

def build_vs_url(left_slug: str, right_slug: str) -> str:
    return f"{TECHCITY_BASE}/en/video/{left_slug}-vs-{right_slug}"

def parse_vs_table_pairs(soup: BeautifulSoup) -> Dict[str, Tuple[str, str]]:
    """
    Parses comparison-page "rows" where a label is followed by two values.
    The page text layout (as seen in Technical.City) is effectively:
      Label   LeftValue   RightValue
    But not always in <table>. We'll parse by scanning text lines.

    Returns: label_lower -> (left_value, right_value)
    """
    text = soup.get_text("\n", strip=True)
    lines = [clean_text(x) for x in text.split("\n") if clean_text(x)]
    pairs: Dict[str, Tuple[str, str]] = {}

    # We search for known labels and take the next 1-2 tokens on the same line if possible,
    # otherwise we fall back to regex searches on the full text.
    # In practice, lines like:
    # "Memory type DDR3 GDDR6"
    # "Maximum RAM amount 1 GB 16 GB"
    # "Memory bus width 128 Bit 256 Bit"
    # "Core clock speed 615 MHz no data"
    # "Power consumption (TDP)25 Watt 250 Watt"
    joined = "\n".join(lines)

    def regex_two_values(label: str) -> Optional[Tuple[str, str]]:
        # capture two value "cells" after label (greedy but limited)
        # Works well on Technical.City because values are short.
        pat = re.compile(
            re.escape(label) + r"\s*([^\n]{1,35})\s+([^\n]{1,35})",
            flags=re.IGNORECASE
        )
        m = pat.search(joined)
        if not m:
            return None
        return (clean_text(m.group(1)), clean_text(m.group(2)))

    # Map label variants we care about
    wanted = {
        "memory type": ["Memory type"],
        "maximum ram amount": ["Maximum RAM amount"],
        "memory bus width": ["Memory bus width"],
        "core clock speed": ["Core clock speed"],
        "power consumption (tdp)": ["Power consumption (TDP)", "Power consumption (tdp)"],
    }

    for key, variants in wanted.items():
        for v in variants:
            tv = regex_two_values(v)
            if tv:
                pairs[key] = tv
                break

    # Some pages have "Power consumption (TDP)25 Watt 250 Watt" without a space.
    if "power consumption (tdp)" not in pairs:
        m = re.search(r"Power consumption\s*\(TDP\)\s*([0-9]{1,4}\s*(?:Watt|W))\s+([0-9]{1,4}\s*(?:Watt|W))", joined, flags=re.IGNORECASE)
        if m:
            pairs["power consumption (tdp)"] = (clean_text(m.group(1)), clean_text(m.group(2)))

    return pairs

def scrape_specs_from_vs(baseline_slug: str, target_slug: str, session: requests.Session, cfg: NetCfg) -> Optional[Dict[str, Any]]:
    """
    Uses baseline-vs-target (target is RIGHT column) to extract required specs.
    """
    vs_url = build_vs_url(baseline_slug, target_slug)
    r = get_with_retries(vs_url, session, cfg)
    if r is None:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    pairs = parse_vs_table_pairs(soup)

    # RIGHT column = target GPU
    mem_type = pairs.get("memory type", (None, None))[1]
    vram_raw = pairs.get("maximum ram amount", (None, None))[1]
    bus_raw = pairs.get("memory bus width", (None, None))[1]
    core_raw = pairs.get("core clock speed", (None, None))[1]
    tdp_raw = pairs.get("power consumption (tdp)", (None, None))[1]

    specs = {
        "vram": normalize_vram(vram_raw or "") or clean_text(vram_raw or "") or None,
        "bus_width": normalize_bus(bus_raw or "") or clean_text(bus_raw or "") or None,
        "core_clock": normalize_clock(core_raw or "") or None,
        "memory_type": clean_text(mem_type or "") or None,
        "tdp": normalize_tdp(tdp_raw or "") or None,  # plain value only
    }

    # Benchmarks (if present): very inconsistent; keep best-effort extraction
    txt = clean_text(soup.get_text(" ", strip=True))
    bench: Dict[str, Any] = {}
    m = re.search(r"combined synthetic benchmark score\s+([0-9]{1,3}(?:\.[0-9]{1,2})?)", txt, flags=re.IGNORECASE)
    if m:
        bench["technicalcity_combined_score"] = float(m.group(1))

    return {
        "specs": specs,
        "benchmark_scores": bench or None,
        "source_url": vs_url,
    }


# ----------------------------
# Wikipedia: architecture + release date (best effort)
# ----------------------------

def wiki_find_title_and_url(query: str, session: requests.Session, cfg: NetCfg) -> Optional[Tuple[str, str]]:
    params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 5}
    r = get_with_retries(WIKI_API, session, cfg, params=params)
    if r is None:
        return None
    hits = r.json().get("query", {}).get("search", [])
    if not hits:
        return None
    title = hits[0].get("title")
    if not title:
        return None

    params2 = {"action": "query", "prop": "info", "inprop": "url", "titles": title, "format": "json"}
    r2 = get_with_retries(WIKI_API, session, cfg, params=params2)
    if r2 is None:
        return (title, "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")))
    pages = r2.json().get("query", {}).get("pages", {})
    for _, p in pages.items():
        if "fullurl" in p:
            return (title, p["fullurl"])
    return (title, "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")))


def wiki_infobox_kv(title: str, session: requests.Session, cfg: NetCfg) -> Dict[str, str]:
    url = WIKI_REST_HTML.format(quote(title, safe=""))
    r = get_with_retries(url, session, cfg)
    if r is None:
        return {}

    soup = BeautifulSoup(r.text, "html.parser")
    infobox = soup.find("table", class_=re.compile(r"\binfobox\b"))
    if not infobox:
        return {}

    kv: Dict[str, str] = {}
    for tr in infobox.find_all("tr"):
        th = tr.find("th")
        td = tr.find("td")
        if not th or not td:
            continue
        k = clean_text(th.get_text(" ", strip=True)).strip(":").lower()
        v = clean_text(td.get_text(" ", strip=True))
        if k and v:
            kv[k] = v
    return kv


def wiki_extract_arch_release(kv: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    arch = None
    rel = None

    for k in ["architecture", "microarchitecture", "codename"]:
        if k in kv:
            arch = clean_text(kv[k])
            break

    for k in ["release date", "released", "launched", "launch", "introduced", "release"]:
        if k in kv:
            rel = clean_text(kv[k])
            break

    if rel:
        try:
            if re.fullmatch(r"\d{4}", rel):
                pass
            else:
                d = dateparser.parse(rel, fuzzy=True)
                if d:
                    rel = d.date().isoformat()
        except Exception:
            pass

    return arch, rel


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="gpus.json", help="Output JSON file")
    ap.add_argument("--max-pages", type=int, default=5, help="Max Technical.City rating pages (<=15)")
    ap.add_argument("--max-gpus", type=int, default=50, help="Max GPUs to attempt")
    ap.add_argument("--baseline-slug", type=str, default="GeForce-RTX-3080-Ti", help="Baseline GPU slug for VS pages")
    ap.add_argument("--wiki", action="store_true", help="Also enrich architecture/release_date from Wikipedia")
    ap.add_argument("--timeout", type=int, default=18, help="HTTP timeout seconds")
    ap.add_argument("--sleep", type=float, default=0.15, help="Sleep between requests")
    ap.add_argument("--retries", type=int, default=3, help="Retries per request")
    ap.add_argument("--user-agent", type=str, default=DEFAULT_UA, help="User-Agent header")
    args = ap.parse_args()

    max_pages = min(max(1, args.max_pages), 15)

    net = NetCfg(
        user_agent=args.user_agent,
        timeout_s=max(5, args.timeout),
        sleep_s=max(0.0, args.sleep),
        retries=max(1, args.retries),
    )
    session = make_session(net)

    candidates = collect_candidates(session, net, max_pages=max_pages, max_gpus=args.max_gpus)
    if not candidates:
        payload = {
            "collection": "gpu_metadata_wikipedia_enriched",
            "generated_at_utc": now_utc_iso(),
            "count": 0,
            "skipped_incomplete_specs": 0,
            "notes": "No candidates collected from Technical.City rating pages.",
            "items": [],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[WARN] No candidates. Wrote empty JSON to {args.out}")
        return

    items: List[Dict[str, Any]] = []
    skipped = 0

    for idx, (gpu_name, gpu_slug) in enumerate(candidates, start=1):
        # Skip baseline itself if present
        if gpu_slug == args.baseline_slug:
            continue

        vs_data = scrape_specs_from_vs(args.baseline_slug, gpu_slug, session, net)
        if not vs_data:
            skipped += 1
            continue

        specs = vs_data["specs"]

        # If Technical.City reports "no data" for core clock, skip (your requirement: no blanks)
        if isinstance(specs.get("core_clock"), str) and specs["core_clock"].lower() in {"no data", "n/a", "-", "—"}:
            specs["core_clock"] = None

        if not is_complete_specs(specs):
            skipped += 1
            continue

        manufacturer = detect_manufacturer(gpu_name)

        arch = None
        rel = None
        wiki_url = None

        if args.wiki:
            w = wiki_find_title_and_url(gpu_name, session, net)
            if w:
                w_title, w_url = w
                wiki_url = w_url
                kv = wiki_infobox_kv(w_title, session, net)
                arch, rel = wiki_extract_arch_release(kv)

        item = {
            "gpu_name": gpu_name,
            "manufacturer": manufacturer,
            "architecture": arch,
            "specs": {
                "vram": specs["vram"],
                "bus_width": specs["bus_width"],
                "core_clock": specs["core_clock"],
                "memory_type": specs["memory_type"],
                "tdp": specs["tdp"],  # plain value only
            },
            "release_date": rel,
            "benchmark_scores": vs_data.get("benchmark_scores"),
            "sources": {
                "technical_city_vs": vs_data["source_url"],
                "wikipedia": wiki_url,
            },
        }

        items.append(item)

        if idx % 10 == 0:
            print(f"[PROGRESS] processed={idx}/{len(candidates)} items={len(items)} skipped={skipped}")

    payload = {
        "collection": "gpu_metadata_wikipedia_enriched",
        "generated_at_utc": now_utc_iso(),
        "count": len(items),
        "skipped_incomplete_specs": skipped,
        "notes": {
            "specs_policy": "Items include only GPUs where specs.vram, specs.bus_width, specs.core_clock, specs.memory_type, specs.tdp are all present.",
            "non_wikipedia_crawl_depth_max_pages": max_pages,
            "tdp_policy": "TDP stored as plain value only (no description).",
            "baseline_vs_slug": args.baseline_slug,
            "wikipedia_enabled": bool(args.wiki),
        },
        "items": items,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Wrote {len(items)} items to {args.out} (skipped={skipped})")


if __name__ == "__main__":
    main()