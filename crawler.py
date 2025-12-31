#!/usr/bin/env python3
"""
GPU Metadata Collector (Wikipedia-first, enrich specs via TechPowerUp + PassMark)

What it collects (per GPU):
✅ GPU Name
✅ Manufacturer (NVIDIA/AMD/Intel)
✅ Architecture
✅ Specs: VRAM, Bus Width, Core Clock, Memory Type, TDP  (NO TDP description)
✅ Release Date
✅ Benchmark scores (if available; PassMark G3D Mark)

Important behavior (per your requirements):
- Wikipedia is the primary source for GPU pages & metadata.
- Specs are NEVER left null/blank in the output:
    * If Wikipedia is missing a spec, we enrich from TechPowerUp GPU DB.
    * If still missing VRAM/TDP, we enrich from PassMark "GPU Mega Page".
    * If a GPU still can't be fully populated (all specs present), it is SKIPPED (not emitted).
- TechPowerUp crawling is limited to MAX=15 pages deep (configurable).

Output:
- JSON file containing only GPUs whose specs are fully populated.
- Includes source URLs used for traceability.

Install:
  pip install requests beautifulsoup4 python-dateutil

Run:
  python gpu_scraper.py --limit 200 --out gpus.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser


WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_BASE = "https://en.wikipedia.org/wiki/"

TPU_LIST_BASE = "https://www.techpowerup.com/gpu-specs/"
TPU_LIST_PAGE = "https://www.techpowerup.com/gpu-specs/"  # listing (paged)
PASSMARK_MEGA = "https://www.videocardbenchmark.net/GPU_mega_page.html"
PASSMARK_LOOKUP = "https://www.videocardbenchmark.net/video_lookup.php"

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
    tpu_max_pages: int  # MAX=15 pages to nested depth


def http_get(url: str, cfg: ScrapeConfig, params: Optional[Dict[str, Any]] = None) -> requests.Response:
    headers = {"User-Agent": cfg.user_agent}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    time.sleep(cfg.sleep_s)
    return r


def mw_api_get(params: Dict[str, Any], cfg: ScrapeConfig) -> Dict[str, Any]:
    r = http_get(WIKI_API, cfg, params=params)
    return r.json()


def clean_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"\[\d+\]", "", s)  # remove [1] citations
    s = re.sub(r"\s+", " ", s).strip()
    return s


def guess_manufacturer(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(x in t for x in ["nvidia", "geforce", "quadro", "tesla", "rtx", "gtx"]):
        return "NVIDIA"
    if any(x in t for x in ["amd", "radeon", "ati", "rx "]):
        return "AMD"
    if any(x in t for x in ["intel", "arc", "iris", "xe"]):
        return "Intel"
    return None


def normalize_gpu_name(name: str) -> str:
    """
    Normalization for matching across sites (simple, conservative).
    """
    n = clean_text(name).lower()
    n = n.replace("®", "").replace("™", "")
    n = re.sub(r"\bgraphics\b", "", n)
    n = re.sub(r"\bvideo card\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def parse_release_date(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    raw = clean_text(raw)
    try:
        # handle pure year
        if re.fullmatch(r"\d{4}", raw):
            return raw
        dt = dateparser.parse(raw, fuzzy=True)
        if dt:
            return dt.date().isoformat()
    except Exception:
        pass
    return raw or None


def extract_infobox_kv(soup: BeautifulSoup) -> Dict[str, str]:
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
            kv[key] = val
    return kv


def first_match_in_kv(kv: Dict[str, str], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in kv and kv[k]:
            return kv[k]
    return None


def parse_wikipedia_html(title: str, cfg: ScrapeConfig) -> Tuple[BeautifulSoup, str]:
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "redirects": 1,
    }
    data = mw_api_get(params, cfg)
    parse = data.get("parse", {})
    html = parse.get("text", {}).get("*", "")
    soup = BeautifulSoup(html, "html.parser")
    url = WIKI_BASE + title.replace(" ", "_")
    return soup, url


def likely_gpu_page(title: str, infobox_kv: Dict[str, str]) -> bool:
    t = title.lower()
    title_signals = ["geforce", "radeon", "quadro", "tesla", "arc", "rtx", "gtx", "rx ", "intel", "nvidia", "amd"]
    kv_signals = ["memory", "tdp", "architecture", "codename", "launch", "release", "bus", "core clock", "gpu clock"]
    return any(s in t for s in title_signals) or any(k in infobox_kv for k in kv_signals)


def get_category_members(category: str, cfg: ScrapeConfig, cap: int) -> List[str]:
    titles: List[str] = []
    cmcontinue: Optional[str] = None
    while len(titles) < cap:
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
        for m in batch:
            if "title" in m:
                titles.append(m["title"])
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return titles[:cap]


def parse_specs_from_wikipedia_infobox(kv: Dict[str, str]) -> Dict[str, Optional[str]]:
    vram = first_match_in_kv(kv, ["memory", "memory size", "vram", "framebuffer", "memory capacity"])
    bus_width = first_match_in_kv(kv, ["bus interface", "memory bus", "bus width"])
    core_clock = first_match_in_kv(kv, ["core clock", "gpu clock", "clock"])
    memory_type = first_match_in_kv(kv, ["memory type", "memory technology"])
    tdp = first_match_in_kv(kv, ["tdp", "power", "thermal design power"])

    def norm_bus(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        s2 = clean_text(s)
        m = re.search(r"(\d{2,4})\s*-\s*bit|(\d{2,4})\s*bit", s2, re.IGNORECASE)
        if m:
            return f"{m.group(1) or m.group(2)}-bit"
        return s2

    def norm_tdp(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        s2 = clean_text(s)
        m = re.search(r"(\d+(?:\.\d+)?)\s*(w|watt|watts)\b", s2, re.IGNORECASE)
        if m:
            return f"{m.group(1)} W"
        return s2

    def norm_vram(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        s2 = clean_text(s)
        m = re.search(r"(\d+(?:\.\d+)?)\s*(gib|gb|mib|mb)\b", s2, re.IGNORECASE)
        if m:
            unit = m.group(2).lower()
            unit = "GB" if unit in ("gib", "gb") else "MB"
            return f"{m.group(1)} {unit}"
        return s2

    def norm_clock(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        s2 = clean_text(s)
        # Keep as-is; GPUs often have base/boost etc.
        return s2

    def norm_memtype(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        return clean_text(s)

    return {
        "vram": norm_vram(vram),
        "bus_width": norm_bus(bus_width),
        "core_clock": norm_clock(core_clock),
        "memory_type": norm_memtype(memory_type),
        "tdp": norm_tdp(tdp),
    }


# ----------------------- TechPowerUp (Specs Enrichment) -----------------------


def crawl_techpowerup_index(cfg: ScrapeConfig) -> Dict[str, str]:
    """
    Crawls up to cfg.tpu_max_pages pages of TechPowerUp GPU DB index and returns:
      normalized_name -> absolute URL to GPU specs page

    Depth limit: MAX=15 pages (per requirement).
    """
    mapping: Dict[str, str] = {}
    for page in range(1, cfg.tpu_max_pages + 1):
        # TPU uses /gpu-specs/ as listing. It can be paged with ?page=N in practice.
        # If it changes, the scraper still remains bounded and fails gracefully.
        params = {"page": page} if page > 1 else None
        try:
            r = http_get(TPU_LIST_PAGE, cfg, params=params)
        except Exception:
            break

        soup = BeautifulSoup(r.text, "html.parser")

        # GPU listing typically contains many links to /gpu-specs/<slug>
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("/gpu-specs/"):
                continue
            text = clean_text(a.get_text(" ", strip=True))
            if not text or len(text) < 3:
                continue
            url = "https://www.techpowerup.com" + href
            key = normalize_gpu_name(text)
            # store first-seen (stable)
            mapping.setdefault(key, url)

        # Small stop condition: if page yields nothing new, likely end or blocked
        if page >= 2 and len(mapping) < 50 * page:
            # heuristic; keep going but likely near end
            pass

    return mapping


def parse_tpu_specs_page(url: str, cfg: ScrapeConfig) -> Dict[str, Optional[str]]:
    """
    Parses relevant fields from a TechPowerUp GPU specs page:
      - VRAM (Memory Size)
      - Bus Width
      - GPU Clock
      - Memory Type
      - TDP
      - Architecture (may be present)
    Returns dict with possible None values (caller will validate/fill).
    """
    r = http_get(url, cfg)
    soup = BeautifulSoup(r.text, "html.parser")

    # TPU pages commonly present a "Specifications" table with field/value rows.
    # We'll parse all table rows (th/td or td/td) into kv for robustness.
    kv: Dict[str, str] = {}

    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            k = clean_text(cells[0].get_text(" ", strip=True)).lower()
            v = clean_text(cells[1].get_text(" ", strip=True))
            if k and v:
                # Avoid overwriting with worse values
                kv.setdefault(k, v)

    def get_any(keys: List[str]) -> Optional[str]:
        for k in keys:
            if k in kv and kv[k]:
                return kv[k]
        return None

    vram = get_any(["memory size", "memory", "memory size (mb)"])
    bus = get_any(["memory bus", "bus width", "memory bus width"])
    clock = get_any(["gpu clock", "core clock", "boost clock"])
    memtype = get_any(["memory type"])
    tdp = get_any(["tdp", "board power", "typical board power"])
    arch = get_any(["architecture", "gpu architecture", "microarchitecture"])

    # Normalize some common patterns
    def norm_vram(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        m = re.search(r"(\d+(?:\.\d+)?)\s*(gb|gib|mb|mib)\b", s, re.IGNORECASE)
        if m:
            unit = m.group(2).lower()
            unit = "GB" if unit in ("gb", "gib") else "MB"
            return f"{m.group(1)} {unit}"
        # Sometimes TPU has MB only number
        m2 = re.search(r"\b(\d{3,6})\b", s)
        if m2:
            return f"{m2.group(1)} MB"
        return s

    def norm_bus(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        m = re.search(r"(\d{2,4})\s*bit", s, re.IGNORECASE)
        if m:
            return f"{m.group(1)}-bit"
        return s

    def norm_tdp(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        m = re.search(r"(\d+(?:\.\d+)?)\s*(w|watt|watts)\b", s, re.IGNORECASE)
        if m:
            return f"{m.group(1)} W"
        return s

    return {
        "vram": norm_vram(vram),
        "bus_width": norm_bus(bus),
        "core_clock": clean_text(clock) if clock else None,
        "memory_type": clean_text(memtype) if memtype else None,
        "tdp": norm_tdp(tdp),
        "architecture": clean_text(arch) if arch else None,
    }


def match_tpu_url(gpu_name: str, tpu_index: Dict[str, str]) -> Optional[str]:
    """
    Fuzzy match a GPU name to a TechPowerUp index entry.
    """
    key = normalize_gpu_name(gpu_name)
    if key in tpu_index:
        return tpu_index[key]

    # fuzzy: compare against a limited candidate set (cheap heuristic)
    best_url = None
    best_score = 0.0
    for k, url in tpu_index.items():
        # quick pruning
        if key and k and key[0] != k[0]:
            continue
        sc = similarity(key, k)
        if sc > best_score:
            best_score = sc
            best_url = url

    # require decent similarity to avoid wrong joins
    return best_url if best_score >= 0.86 else None


# ----------------------- PassMark (Benchmarks + VRAM/TDP fallback) -----------------------


def load_passmark_mega(cfg: ScrapeConfig) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Loads PassMark GPU mega page (single page) and builds:
      normalized_name -> {"g3d_mark": "...", "tdp": "... W", "vram": "... MB", "source_url": "..."}
    """
    r = http_get(PASSMARK_MEGA, cfg)
    soup = BeautifulSoup(r.text, "html.parser")

    # Find the largest table on the page
    tables = soup.find_all("table")
    if not tables:
        return {}

    # Heuristic: choose table with most rows
    big = max(tables, key=lambda t: len(t.find_all("tr")))
    rows = big.find_all("tr")
    if len(rows) < 2:
        return {}

    # Build column index from header
    header_cells = [clean_text(c.get_text(" ", strip=True)).lower() for c in rows[0].find_all(["th", "td"])]
    # common columns: "videocard name", "g3d mark", "tdp (w)", "vram (mb)" etc.
    def col_idx(possible: List[str]) -> Optional[int]:
        for p in possible:
            for i, h in enumerate(header_cells):
                if p in h:
                    return i
        return None

    idx_name = col_idx(["videocard name", "video card", "videocard"])
    idx_g3d = col_idx(["g3d mark"])
    idx_tdp = col_idx(["tdp"])
    idx_vram = col_idx(["vram"])

    if idx_name is None:
        return {}

    out: Dict[str, Dict[str, Optional[str]]] = {}

    for tr in rows[1:]:
        cells = tr.find_all(["th", "td"])
        if len(cells) <= idx_name:
            continue
        name_a = cells[idx_name].find("a", href=True)
        name = clean_text(cells[idx_name].get_text(" ", strip=True))
        if not name:
            continue

        g3d = clean_text(cells[idx_g3d].get_text(" ", strip=True)) if idx_g3d is not None and len(cells) > idx_g3d else None
        tdp = clean_text(cells[idx_tdp].get_text(" ", strip=True)) if idx_tdp is not None and len(cells) > idx_tdp else None
        vram = clean_text(cells[idx_vram].get_text(" ", strip=True)) if idx_vram is not None and len(cells) > idx_vram else None

        # normalize g3d as integer string when possible
        if g3d:
            m = re.search(r"\d[\d,]*", g3d)
            g3d = m.group(0).replace(",", "") if m else g3d

        # normalize tdp to "X W"
        if tdp:
            m = re.search(r"(\d+(?:\.\d+)?)", tdp)
            tdp = f"{m.group(1)} W" if m else None

        # normalize vram to "... MB"
        if vram:
            m = re.search(r"(\d+)", vram)
            vram = f"{m.group(1)} MB" if m else None

        src_url = None
        if name_a:
            href = name_a["href"]
            if href.startswith("/"):
                src_url = "https://www.videocardbenchmark.net" + href
            elif href.startswith("http"):
                src_url = href

        out[normalize_gpu_name(name)] = {
            "g3d_mark": g3d,
            "tdp": tdp,
            "vram": vram,
            "source_url": src_url or PASSMARK_MEGA,
        }

    return out


def passmark_lookup_g3d(gpu_name: str, cfg: ScrapeConfig) -> Optional[Dict[str, Any]]:
    """
    Uses PassMark lookup endpoint to find the GPU and (if possible) extract the numeric G3D Mark
    by following to gpu.php?...&id=...
    Returns {"g3d_mark": "....", "source_url": "..."} or None
    """
    params = {"gpu": gpu_name}
    r = http_get(PASSMARK_LOOKUP, cfg, params=params)
    soup = BeautifulSoup(r.text, "html.parser")

    # Often the page contains a link to gpu.php?gpu=...&id=...
    link = soup.find("a", href=re.compile(r"/gpu\.php\?gpu="))
    if not link:
        return None

    href = link.get("href", "")
    url = "https://www.videocardbenchmark.net" + href if href.startswith("/") else href
    try:
        rr = http_get(url, cfg)
    except Exception:
        return {"g3d_mark": None, "source_url": url}

    ss = BeautifulSoup(rr.text, "html.parser")

    # Look for "Average G3D Mark:" label
    text = clean_text(ss.get_text(" ", strip=True))
    m = re.search(r"average g3d mark[:\s]+(\d[\d,]*)", text, re.IGNORECASE)
    if m:
        return {"g3d_mark": m.group(1).replace(",", ""), "source_url": url}

    return {"g3d_mark": None, "source_url": url}


# ----------------------- Main GPU Extraction + Enrichment -----------------------


def has_all_specs(specs: Dict[str, Optional[str]]) -> bool:
    required = ["vram", "bus_width", "core_clock", "memory_type", "tdp"]
    return all(specs.get(k) and clean_text(str(specs.get(k))) for k in required)


def extract_gpu_from_wikipedia(title: str, cfg: ScrapeConfig) -> Optional[Dict[str, Any]]:
    soup, wiki_url = parse_wikipedia_html(title, cfg)
    kv = extract_infobox_kv(soup)
    if not likely_gpu_page(title, kv):
        return None

    gpu_name = clean_text(title)

    manufacturer_raw = first_match_in_kv(kv, ["manufacturer", "vendor", "developed by", "developer", "produced by"])
    manufacturer = guess_manufacturer(manufacturer_raw or gpu_name) or "Unknown"

    architecture = first_match_in_kv(kv, ["architecture", "microarchitecture", "codename", "core"])
    release_raw = first_match_in_kv(kv, ["release date", "launched", "launch", "introduced", "release"])
    release_date = parse_release_date(release_raw)

    specs = parse_specs_from_wikipedia_infobox(kv)

    return {
        "gpu_name": gpu_name,
        "manufacturer": manufacturer,
        "architecture": clean_text(architecture) if architecture else None,
        "specs": specs,  # may include None; will be enriched
        "release_date": release_date,
        "benchmarks": None,  # will be enriched
        "sources": {"wikipedia": wiki_url},
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def enrich_specs(
    entry: Dict[str, Any],
    cfg: ScrapeConfig,
    tpu_index: Dict[str, str],
    passmark_mega: Dict[str, Dict[str, Optional[str]]],
) -> Tuple[Dict[str, Any], bool]:
    """
    Enriches entry in-place and returns (entry, ok_fully_populated).
    """
    gpu_name = entry["gpu_name"]
    sources = entry.setdefault("sources", {})

    # 1) TechPowerUp enrichment for missing specs (and architecture if missing)
    tpu_url = match_tpu_url(gpu_name, tpu_index)
    if tpu_url:
        try:
            tpu_specs = parse_tpu_specs_page(tpu_url, cfg)
            sources.setdefault("techpowerup", tpu_url)

            # Fill architecture if missing
            if not entry.get("architecture") and tpu_specs.get("architecture"):
                entry["architecture"] = tpu_specs["architecture"]

            # Fill missing spec fields
            for k in ["vram", "bus_width", "core_clock", "memory_type", "tdp"]:
                if not entry["specs"].get(k) and tpu_specs.get(k):
                    entry["specs"][k] = tpu_specs[k]
        except Exception:
            pass

    # 2) PassMark mega fallback for VRAM + TDP if still missing
    pm_key = normalize_gpu_name(gpu_name)
    pm_row = passmark_mega.get(pm_key)
    if not pm_row:
        # fuzzy try (low-cost): scan a few near matches
        best = None
        best_score = 0.0
        for k, row in passmark_mega.items():
            sc = similarity(pm_key, k)
            if sc > best_score:
                best_score = sc
                best = row
        if best_score >= 0.90:
            pm_row = best

    if pm_row:
        sources.setdefault("passmark_mega", pm_row.get("source_url") or PASSMARK_MEGA)
        if not entry["specs"].get("vram") and pm_row.get("vram"):
            entry["specs"]["vram"] = pm_row["vram"]
        if not entry["specs"].get("tdp") and pm_row.get("tdp"):
            entry["specs"]["tdp"] = pm_row["tdp"]

    # Final validation: specs must be complete
    ok = has_all_specs(entry["specs"])
    return entry, ok


def enrich_benchmarks(entry: Dict[str, Any], cfg: ScrapeConfig, passmark_mega: Dict[str, Dict[str, Optional[str]]]) -> None:
    """
    Adds PassMark G3D Mark when available (best-effort).
    """
    gpu_name = entry["gpu_name"]
    sources = entry.setdefault("sources", {})

    # Prefer mega page mapping (fast)
    pm_key = normalize_gpu_name(gpu_name)
    row = passmark_mega.get(pm_key)
    if row and row.get("g3d_mark"):
        entry["benchmarks"] = {"passmark_g3d_mark": row["g3d_mark"]}
        sources.setdefault("passmark_mega", row.get("source_url") or PASSMARK_MEGA)
        return

    # Otherwise try lookup (slower)
    try:
        res = passmark_lookup_g3d(gpu_name, cfg)
        if res and res.get("g3d_mark"):
            entry["benchmarks"] = {"passmark_g3d_mark": res["g3d_mark"]}
            sources.setdefault("passmark_lookup", res.get("source_url"))
        else:
            entry["benchmarks"] = None
    except Exception:
        entry["benchmarks"] = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="Max number of GPUs to output (fully populated specs only)")
    ap.add_argument("--out", type=str, default="gpus.json", help="Output JSON file")
    ap.add_argument("--sleep", type=float, default=0.25, help="Sleep between requests (seconds)")
    ap.add_argument(
        "--user-agent",
        type=str,
        default="GPUMetadataCollector/2.0 (contact: you@example.com)",
        help="Custom User-Agent (recommended by sites).",
    )
    ap.add_argument("--tpu-max-pages", type=int, default=15, help="MAX pages to crawl in TechPowerUp index (<=15 recommended)")
    ap.add_argument("--candidate-cap", type=int, default=1200, help="How many Wikipedia category members to consider before filtering")
    args = ap.parse_args()

    cfg = ScrapeConfig(
        limit=args.limit,
        sleep_s=args.sleep,
        user_agent=args.user_agent,
        tpu_max_pages=min(args.tpu_max_pages, 15),  # enforce <=15
    )

    # Preload enrichment sources (bounded)
    print("[INFO] Crawling TechPowerUp index (bounded)...")
    tpu_index = crawl_techpowerup_index(cfg)
    print(f"[INFO] TechPowerUp index entries: {len(tpu_index)}")

    print("[INFO] Loading PassMark GPU mega page...")
    passmark_mega = load_passmark_mega(cfg)
    print(f"[INFO] PassMark mega entries: {len(passmark_mega)}")

    # Collect candidate Wikipedia pages
    seen = set()
    candidates: List[str] = []
    for cat in GPU_CATEGORIES:
        for title in get_category_members(cat, cfg, cap=args.candidate_cap):
            if title not in seen:
                seen.add(title)
                candidates.append(title)

    results: List[Dict[str, Any]] = []
    skipped_incomplete = 0

    for title in candidates:
        if len(results) >= cfg.limit:
            break

        try:
            entry = extract_gpu_from_wikipedia(title, cfg)
            if not entry:
                continue

            entry, ok = enrich_specs(entry, cfg, tpu_index, passmark_mega)
            if not ok:
                skipped_incomplete += 1
                continue

            # Benchmarks (best-effort)
            enrich_benchmarks(entry, cfg, passmark_mega)

            # Ensure manufacturer is one of NVIDIA/AMD/Intel when possible
            if entry["manufacturer"] == "Unknown":
                entry["manufacturer"] = guess_manufacturer(entry["gpu_name"]) or "Unknown"

            results.append(entry)
            print(f"[OK] {entry['gpu_name']} (specs complete)")

        except requests.HTTPError as e:
            print(f"[HTTP] {title}: {e}")
        except Exception as e:
            print(f"[ERR] {title}: {e}")

    payload = {
        "collection": "gpu_metadata_wikipedia_enriched",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(results),
        "skipped_incomplete_specs": skipped_incomplete,
        "notes": {
            "specs_policy": "Output includes only GPUs where specs.vram, specs.bus_width, specs.core_clock, specs.memory_type, specs.tdp are all present.",
            "techpowerup_index_pages_crawled_max": cfg.tpu_max_pages,
            "tdp_policy": "TDP is stored as a plain value only (no description).",
        },
        "items": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Wrote {len(results)} GPUs to {args.out}")
    print(f"[INFO] Skipped due to incomplete specs after enrichment: {skipped_incomplete}")


if __name__ == "__main__":
    main()