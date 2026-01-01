import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from flask import Flask, request, jsonify, render_template_string

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS


DEFAULT_TTL_PATH = os.environ.get("KG_TTL_PATH", "/mnt/data/e27afcf6-5f87-40ad-b2e8-01bc625f6f0c.ttl")
DEFAULT_ENTITIES_JSON_PATH = os.environ.get("KG_ENTITIES_JSON_PATH", "/mnt/data/60033a0d-87f9-4756-8862-029bf897c1e4.json")
DEFAULT_RELATIONS_JSON_PATH = os.environ.get("KG_RELATIONS_JSON_PATH", "/mnt/data/a299efd0-2cac-47dc-a02c-d0c058df0cbb.json")

DEFAULT_OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

app = Flask(__name__)

KB_LOCK = None


@dataclass
class Entity:
    id: str
    name: str
    etype: str
    category: str
    source_text: str
    source_url: str


@dataclass
class Relation:
    relation_id: str
    source_id: str
    predicate: str
    target_id: str
    source_text: str
    source_url: str


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _pretty_local_name(s: str) -> str:
    if "#" in s:
        return s.split("#")[-1]
    if "/" in s:
        return s.rstrip("/").split("/")[-1]
    return s


def _extract_numbers(text: str) -> Set[str]:
    return set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text))


def _tokenize(text: str) -> List[str]:
    t = re.sub(r"[^a-zA-Z0-9:_\-]+", " ", text.lower()).strip()
    return [x for x in t.split() if x]


def _looks_like_uri(s: str) -> bool:
    if not s:
        return False
    if " " in s or "\n" in s or "\t" in s:
        return False
    if s.startswith("http://") or s.startswith("https://") or s.startswith("urn:"):
        return True
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:[^\s]+$", s):
        return True
    return False


def _is_pathlike(p: str) -> bool:
    if not p:
        return False
    if p.startswith("/") or p.startswith("./") or p.startswith("../"):
        return True
    if re.match(r"^[A-Za-z]:\\", p):
        return True
    return False


def parse_gpu_specs_from_text(source_text: str) -> Dict[str, Any]:
    s = source_text
    out: Dict[str, Any] = {}
    m = re.search(r"VRAM\s+(\d+(?:\.\d+)?)\s*GB", s, re.IGNORECASE)
    if m:
        out["vram_gb"] = float(m.group(1))
    m = re.search(r"Bus\s+(\d+)\s*bit", s, re.IGNORECASE)
    if m:
        out["bus_bit"] = int(m.group(1))
    m = re.search(r"Core\s+(\d+(?:\.\d+)?)\s*MHz", s, re.IGNORECASE)
    if m:
        out["core_mhz"] = float(m.group(1))
    m = re.search(r"Memory\s+([A-Za-z0-9]+)", s, re.IGNORECASE)
    if m:
        out["memory_type_text"] = m.group(1)
    m = re.search(r"TDP\s+(\d+(?:\.\d+)?)\s*W", s, re.IGNORECASE)
    if m:
        out["tdp_w"] = float(m.group(1))
    return out


def parse_query_constraints(q: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m = re.search(r"(\d+(?:\.\d+)?)\s*gb", q, re.IGNORECASE)
    if m:
        out["min_vram_gb"] = float(m.group(1))

    m = re.search(r"tdp\s*(?:<=|<|at\s*most|max(?:imum)?)\s*(\d+(?:\.\d+)?)\s*w", q, re.IGNORECASE)
    if m:
        out["max_tdp_w"] = float(m.group(1))
    else:
        if re.search(r"(<=|<|at\s*most|max)", q, re.IGNORECASE):
            m2 = re.search(r"(\d+(?:\.\d+)?)\s*w", q, re.IGNORECASE)
            if m2:
                out["max_tdp_w"] = float(m2.group(1))

    mem = None
    for k in ["gddr7", "gddr6x", "gddr6", "hbm3", "hbm2e", "hbm2"]:
        if re.search(r"\b" + re.escape(k) + r"\b", q, re.IGNORECASE):
            mem = k.upper()
            break
    if mem:
        out["memory_type"] = mem

    arch = None
    for k in ["blackwell", "ada", "ada lovelace", "rdna 3", "rdna3", "hopper", "ampere", "turing"]:
        if re.search(r"\b" + re.escape(k) + r"\b", q, re.IGNORECASE):
            arch = k
            break
    if arch:
        out["architecture_hint"] = arch

    mfg = None
    for k in ["nvidia", "amd", "intel"]:
        if re.search(r"\b" + re.escape(k) + r"\b", q, re.IGNORECASE):
            mfg = k.upper()
            break
    if mfg:
        out["manufacturer"] = mfg

    return out


class KnowledgeBase:
    def __init__(self) -> None:
        self.g = Graph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.node_text_index: Dict[str, str] = {}
        self.node_tokens: Dict[str, Set[str]] = {}
        self.gpu_specs: Dict[str, Dict[str, Any]] = {}
        self.loaded_ttl_path: Optional[str] = None
        self.loaded_entities_json_path: Optional[str] = None
        self.loaded_relations_json_path: Optional[str] = None

    def load_ttl(self, path: str) -> None:
        self.g = Graph()
        self.g.parse(path, format="turtle")
        self.loaded_ttl_path = path

    def load_entities_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entities = {}
        self.gpu_specs = {}
        for item in data:
            e = Entity(
                id=_safe_str(item.get("id")),
                name=_safe_str(item.get("entity_name")),
                etype=_safe_str(item.get("entity_type")),
                category=_safe_str(item.get("category")),
                source_text=_safe_str(item.get("source_text")),
                source_url=_safe_str(item.get("source_url")),
            )
            if e.id:
                self.entities[e.id] = e
                if e.etype.lower() == "gpu":
                    self.gpu_specs[e.id] = parse_gpu_specs_from_text(e.source_text)
        self.loaded_entities_json_path = path

    def load_relations_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.relations = []
        for item in data:
            r = Relation(
                relation_id=_safe_str(item.get("relation_id")),
                source_id=_safe_str(item.get("source_id")),
                predicate=_safe_str(item.get("predicate")),
                target_id=_safe_str(item.get("target_id")),
                source_text=_safe_str(item.get("source_text")),
                source_url=_safe_str(item.get("source_url")),
            )
            if r.source_id and r.target_id:
                self.relations.append(r)
        self.loaded_relations_json_path = path

    def build_index(self) -> None:
        self.node_text_index = {}
        self.node_tokens = {}

        for s in set([_safe_str(x) for x in self.g.subjects()] + [_safe_str(x) for x in self.g.objects()]):
            label = self.get_label_for_node_str(s)
            self.node_text_index[s] = (self.node_text_index.get(s, "") + " " + label).strip()

        for eid, e in self.entities.items():
            self.node_text_index[eid] = (
                self.node_text_index.get(eid, "")
                + " "
                + e.name
                + " "
                + e.source_text
                + " "
                + e.etype
                + " "
                + e.category
            ).strip()

        for nid, text in self.node_text_index.items():
            self.node_tokens[nid] = set(_tokenize(text))

    def get_label_for_node_str(self, node_str: str) -> str:
        if node_str in self.entities:
            return self.entities[node_str].name
        if _looks_like_uri(node_str):
            try:
                node = URIRef(node_str)
                for lbl in self.g.objects(node, RDFS.label):
                    if isinstance(lbl, Literal):
                        return _safe_str(lbl)
            except Exception:
                pass
        return _pretty_local_name(node_str)

    def get_triples_for_node(self, node_str: str) -> List[Tuple[str, str, str]]:
        if not _looks_like_uri(node_str):
            return []
        triples: List[Tuple[str, str, str]] = []
        node = URIRef(node_str)
        for s, p, o in self.g.triples((node, None, None)):
            triples.append((_safe_str(s), _safe_str(p), _safe_str(o)))
        for s, p, o in self.g.triples((None, None, node)):
            triples.append((_safe_str(s), _safe_str(p), _safe_str(o)))
        return triples

    def neighbors(self, node_str: str) -> Set[str]:
        out: Set[str] = set()
        if _looks_like_uri(node_str):
            node = URIRef(node_str)
            for s, p, o in self.g.triples((node, None, None)):
                out.add(_safe_str(o))
            for s, p, o in self.g.triples((None, None, node)):
                out.add(_safe_str(s))
        for r in self.relations:
            if r.source_id == node_str:
                out.add(r.target_id)
            if r.target_id == node_str:
                out.add(r.source_id)
        return out


@dataclass
class EvidenceLine:
    raw: str
    s_id: str
    p: str
    o_id: str
    s_label: str
    o_label: str
    kind: str
    source_url: str


@dataclass
class RetrievalResult:
    candidate_nodes: List[str]
    evidence_triples: List[Tuple[str, str, str]]
    evidence_relations: List[Tuple[str, str, str, str, str]]
    context_pack_text: str
    trace_raw: Dict[str, List[str]]
    trace_pretty: Dict[str, List[EvidenceLine]]


@dataclass
class RankedItem:
    item_id: str
    item_name: str
    score: float
    reasons: List[str]
    evidence_pretty: List[EvidenceLine]
    source_urls: List[str]


def retrieve_subgraph(kb: KnowledgeBase, query: str, max_seeds: int = 30, depth: int = 2, max_triples: int = 250) -> RetrievalResult:
    q_tokens = set(_tokenize(query))
    scored: List[Tuple[float, str]] = []
    for nid, toks in kb.node_tokens.items():
        inter = len(q_tokens & toks)
        if inter == 0:
            continue
        union = len(q_tokens | toks)
        j = inter / union if union else 0.0
        scored.append((j, nid))
    scored.sort(reverse=True, key=lambda x: x[0])

    seeds = [nid for _, nid in scored[:max_seeds]]
    visited: Set[str] = set(seeds)
    frontier: Set[str] = set(seeds)

    for _ in range(depth):
        nxt: Set[str] = set()
        for n in frontier:
            for nb in kb.neighbors(n):
                if nb not in visited:
                    visited.add(nb)
                    nxt.add(nb)
        frontier = nxt
        if not frontier:
            break

    nodes = list(visited)
    triples: List[Tuple[str, str, str]] = []
    rels: List[Tuple[str, str, str, str, str]] = []

    for n in nodes:
        triples.extend(kb.get_triples_for_node(n))

    seen_rel: Set[Tuple[str, str, str]] = set()
    for r in kb.relations:
        if r.source_id in visited and r.target_id in visited:
            key = (r.source_id, r.predicate, r.target_id)
            if key not in seen_rel:
                seen_rel.add(key)
                rels.append((r.source_id, r.predicate, r.target_id, r.source_text, r.source_url))

    triples = list(dict.fromkeys(triples))
    if len(triples) > max_triples:
        triples = triples[:max_triples]

    trace_raw: Dict[str, List[str]] = {}
    trace_pretty: Dict[str, List[EvidenceLine]] = {}

    def add_trace(s_id: str, p: str, o_id: str, kind: str, source_url: str) -> None:
        raw = f"{s_id} {p} {o_id}"
        s_label = kb.get_label_for_node_str(s_id)
        o_label = kb.get_label_for_node_str(o_id)
        el = EvidenceLine(
            raw=raw,
            s_id=s_id,
            p=p,
            o_id=o_id,
            s_label=s_label,
            o_label=o_label,
            kind=kind,
            source_url=source_url,
        )
        trace_raw.setdefault(s_id, []).append(raw)
        trace_raw.setdefault(o_id, []).append(raw)
        trace_pretty.setdefault(s_id, []).append(el)
        trace_pretty.setdefault(o_id, []).append(el)

    for (s, p, o) in triples:
        add_trace(s, p, o, "ttl", "")

    for (s, p, o, st, su) in rels:
        add_trace(s, p, o, "json", su or "")

    context_lines: List[str] = []
    context_lines.append("EVIDENCE_TRIPLES:")
    for (s, p, o) in triples:
        context_lines.append(f"- {kb.get_label_for_node_str(s)} | {_pretty_local_name(p)} | {kb.get_label_for_node_str(o)}")

    if rels:
        context_lines.append("EVIDENCE_RELATIONS_JSON:")
        for (s, p, o, st, su) in rels:
            context_lines.append(f"- {kb.get_label_for_node_str(s)} | {p} | {kb.get_label_for_node_str(o)} | src: {su}")

    attrs_lines: List[str] = []
    gpu_candidates = [n for n in nodes if (n in kb.entities and kb.entities[n].etype.lower() == "gpu")]
    if gpu_candidates:
        attrs_lines.append("GPU_ATTRIBUTES_FROM_JSON:")
        for gid in gpu_candidates[:120]:
            e = kb.entities.get(gid)
            if not e:
                continue
            specs = kb.gpu_specs.get(gid, {})
            attrs_lines.append(f"- {e.name} ({gid}) specs: {json.dumps(specs, ensure_ascii=False)}")

    context_pack = "\n".join(context_lines + attrs_lines)

    return RetrievalResult(
        candidate_nodes=nodes,
        evidence_triples=triples,
        evidence_relations=rels,
        context_pack_text=context_pack,
        trace_raw=trace_raw,
        trace_pretty=trace_pretty,
    )


def rank_technologies(kb: KnowledgeBase, retrieval: RetrievalResult, query: str, top_k: int = 10) -> List[RankedItem]:
    constraints = parse_query_constraints(query)

    candidates: List[str] = []
    for n in retrieval.candidate_nodes:
        if n in kb.entities and kb.entities[n].etype.lower() == "gpu":
            candidates.append(n)

    seen: Set[str] = set()
    candidates_unique: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            candidates_unique.append(c)

    scored: List[RankedItem] = []
    for cid in candidates_unique:
        e = kb.entities.get(cid)
        if not e:
            continue

        specs = kb.gpu_specs.get(cid, {})
        score = 0.0
        reasons: List[str] = []
        urls: Set[str] = set()
        ev_pretty: List[EvidenceLine] = []

        if e.source_url:
            urls.add(e.source_url)

        if "min_vram_gb" in constraints and "vram_gb" in specs:
            if specs["vram_gb"] >= constraints["min_vram_gb"]:
                score += 3.0
                reasons.append(f"VRAM {specs['vram_gb']} GB meets >= {constraints['min_vram_gb']} GB")
            else:
                score -= 2.5
                reasons.append(f"VRAM {specs['vram_gb']} GB below {constraints['min_vram_gb']} GB")

        if "max_tdp_w" in constraints and "tdp_w" in specs:
            if specs["tdp_w"] <= constraints["max_tdp_w"]:
                score += 2.0
                reasons.append(f"TDP {specs['tdp_w']} W meets <= {constraints['max_tdp_w']} W")
            else:
                score -= 1.5
                reasons.append(f"TDP {specs['tdp_w']} W above {constraints['max_tdp_w']} W")

        if "memory_type" in constraints:
            mt_q = constraints["memory_type"]
            mt_s = _safe_str(specs.get("memory_type_text", "")).upper()
            if mt_s and mt_q == mt_s:
                score += 2.0
                reasons.append(f"Memory {mt_s} matches {mt_q}")
            elif mt_s:
                score -= 0.5
                reasons.append(f"Memory {mt_s} does not match {mt_q}")

        ev_pretty.extend(retrieval.trace_pretty.get(cid, []))
        for ev in ev_pretty:
            if ev.source_url:
                urls.add(ev.source_url)

        if constraints.get("manufacturer"):
            mfg = constraints["manufacturer"]
            mfg_match = False
            for (s, pred, o, st, su) in retrieval.evidence_relations:
                if o == cid and pred.lower() == "produces" and s in kb.entities:
                    if kb.entities[s].name.strip().upper() == mfg:
                        mfg_match = True
                        if su:
                            urls.add(su)
            if mfg_match:
                score += 1.5
                reasons.append(f"Manufacturer matches {mfg}")
            else:
                score -= 0.2
                reasons.append(f"Manufacturer not confirmed as {mfg} in retrieved evidence")

        if constraints.get("architecture_hint"):
            ah = constraints["architecture_hint"].lower()
            arch_match = False
            for (s, pred, o, st, su) in retrieval.evidence_relations:
                if s == cid and pred.lower() == "hasarchitecture" and o in kb.entities:
                    if ah in kb.entities[o].name.lower():
                        arch_match = True
                        if su:
                            urls.add(su)
            if arch_match:
                score += 1.2
                reasons.append("Architecture matches query hint")
            else:
                reasons.append("Architecture not confirmed as query hint in retrieved evidence")

        if not ev_pretty:
            ev_pretty = [EvidenceLine(raw=cid, s_id=cid, p="", o_id="", s_label=e.name, o_label="", kind="id", source_url=e.source_url or "")]

        scored.append(
            RankedItem(
                item_id=cid,
                item_name=e.name,
                score=score,
                reasons=reasons,
                evidence_pretty=ev_pretty[:30],
                source_urls=sorted([u for u in urls if u])[:12],
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        url = self.base_url + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return _safe_str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))


def build_llm_prompt(query: str, ranked: List[RankedItem], context_pack: str) -> Tuple[str, str]:
    sys = (
        "You are a constrained assistant. You must ONLY use the provided EVIDENCE to produce output.\n"
        "Rules:\n"
        "1) Do not introduce any fact, number, comparison, or attribute not explicitly present in EVIDENCE.\n"
        "2) If EVIDENCE is insufficient for any part, write 'INSUFFICIENT_EVIDENCE'.\n"
        "3) You must cite trace references using evidence lines exactly.\n"
        "4) Output must be deterministic and grounded.\n"
        "5) Do not add creativity.\n"
    )
    ranked_block = "\n".join([f"- {i.item_name} ({i.item_id}) score={i.score:.3f}" for i in ranked])
    user = (
        f"USER_QUERY:\n{query}\n\n"
        f"RANKED_CANDIDATES:\n{ranked_block}\n\n"
        f"EVIDENCE:\n{context_pack}\n\n"
        "TASK:\n"
        "Return a ranked recommendation list aligned with RANKED_CANDIDATES.\n"
        "For each item: provide a short explanation composed only of EVIDENCE facts.\n"
        "For each explanation: include a TRACE section listing the exact evidence lines you used.\n"
        "If any explanation would require missing info, write INSUFFICIENT_EVIDENCE.\n"
    )
    return sys, user


def validate_llm_output(evidence_context: str, llm_text: str) -> bool:
    ev_nums = _extract_numbers(evidence_context)
    out_nums = _extract_numbers(llm_text)
    if not out_nums:
        return True
    for n in out_nums:
        if n not in ev_nums:
            return False
    return True


def kb_stats(kb: KnowledgeBase) -> Dict[str, Any]:
    n_triples = len(kb.g)
    n_entities = len(kb.entities)
    n_relations = len(kb.relations)
    types: Dict[str, int] = {}
    cats: Dict[str, int] = {}
    for e in kb.entities.values():
        t = e.etype or "unknown"
        c = e.category or "unknown"
        types[t] = types.get(t, 0) + 1
        cats[c] = cats.get(c, 0) + 1
    return {
        "ttl_path": kb.loaded_ttl_path,
        "entities_json_path": kb.loaded_entities_json_path,
        "relations_json_path": kb.loaded_relations_json_path,
        "triple_count": n_triples,
        "entity_count": n_entities,
        "relation_count": n_relations,
        "entity_types": dict(sorted(types.items(), key=lambda x: (-x[1], x[0])))[:50],
        "categories": dict(sorted(cats.items(), key=lambda x: (-x[1], x[0])))[:50],
    }


GLOBAL_KB = KnowledgeBase()
GLOBAL_STATS: Dict[str, Any] = {}
GLOBAL_LOAD_ERR: str = ""


def load_kb(ttl_path: str, entities_json_path: str, relations_json_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    global GLOBAL_KB, GLOBAL_STATS, GLOBAL_LOAD_ERR
    try:
        kb = KnowledgeBase()
        if ttl_path and os.path.exists(ttl_path):
            kb.load_ttl(ttl_path)
        else:
            if ttl_path:
                return False, f"TTL path not found: {ttl_path}", {}
        if entities_json_path and os.path.exists(entities_json_path):
            kb.load_entities_json(entities_json_path)
        else:
            if entities_json_path:
                return False, f"Entities JSON path not found: {entities_json_path}", {}
        if relations_json_path and os.path.exists(relations_json_path):
            kb.load_relations_json(relations_json_path)
        else:
            if relations_json_path:
                return False, f"Relations JSON path not found: {relations_json_path}", {}

        kb.build_index()
        GLOBAL_KB = kb
        GLOBAL_STATS = kb_stats(kb)
        GLOBAL_LOAD_ERR = ""
        return True, "", GLOBAL_STATS
    except Exception as ex:
        GLOBAL_LOAD_ERR = _safe_str(ex)
        return False, GLOBAL_LOAD_ERR, {}


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>KG-Grounded RAG Recommender (Web)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { --bg:#0b0f17; --card:#121a2a; --muted:#9db0d0; --text:#e8eefc; --accent:#6aa9ff; --danger:#ff6a6a; --ok:#63e6be; }
    body { background:var(--bg); color:var(--text); font-family:ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin:0; padding:18px; }
    .grid { display:grid; grid-template-columns: 1.2fr 1fr; gap:14px; }
    @media (max-width: 1100px) { .grid { grid-template-columns:1fr; } }
    .card { background:var(--card); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:14px; }
    h1 { margin:0 0 10px 0; font-size:20px; }
    h2 { margin:0 0 10px 0; font-size:16px; color:var(--muted); font-weight:600; }
    label { display:block; font-size:12px; color:var(--muted); margin:10px 0 6px; }
    input, textarea, select { width:100%; box-sizing:border-box; background:#0f1626; color:var(--text); border:1px solid rgba(255,255,255,0.10); border-radius:10px; padding:10px; outline:none; }
    textarea { min-height:110px; resize:vertical; }
    .row { display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
    @media (max-width: 700px) { .row { grid-template-columns:1fr; } }
    button { background:var(--accent); color:#051126; border:none; padding:10px 12px; border-radius:10px; cursor:pointer; font-weight:700; }
    button.secondary { background:#23314d; color:var(--text); border:1px solid rgba(255,255,255,0.12); }
    button.danger { background:var(--danger); color:#1b0b0b; }
    .muted { color:var(--muted); font-size:12px; }
    .pill { display:inline-block; padding:4px 8px; border-radius:999px; background:#223252; color:var(--muted); font-size:12px; margin-right:6px; border:1px solid rgba(255,255,255,0.10); }
    .ok { color:var(--ok); font-weight:700; }
    .err { color:var(--danger); font-weight:700; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size:12px; }
    .item { border:1px solid rgba(255,255,255,0.10); border-radius:12px; padding:12px; margin:10px 0; background:rgba(0,0,0,0.12); }
    .item h3 { margin:0 0 8px 0; font-size:15px; }
    .kv { display:grid; grid-template-columns: 140px 1fr; gap:8px; margin:6px 0; }
    .kv div:first-child { color:var(--muted); font-size:12px; }
    details { border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:10px; margin-top:10px; background:rgba(0,0,0,0.10); }
    summary { cursor:pointer; color:var(--muted); font-weight:700; }
    .smallbtns { display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
    .topbar { display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:14px; }
    .right { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    .status { font-size:12px; color:var(--muted); }
    .nowrap { white-space:nowrap; }
  </style>
</head>
<body>
  <div class="topbar">
    <div>
      <h1>KG-Grounded RAG Recommender (Traceable)</h1>
      <div class="status">
        <span class="pill nowrap">TTL triples: <span id="st_triples">-</span></span>
        <span class="pill nowrap">Entities: <span id="st_entities">-</span></span>
        <span class="pill nowrap">Relations: <span id="st_relations">-</span></span>
        <span class="pill nowrap">Loaded: <span id="st_loaded" class="muted">no</span></span>
      </div>
    </div>
    <div class="right">
      <button class="secondary" id="btnReload">Reload KG</button>
      <button class="secondary" id="btnStats">Refresh Stats</button>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Inputs</h2>

      <label>TTL path</label>
      <input id="ttl_path" value="{{ ttl_path|e }}"/>

      <label>Entities JSON path</label>
      <input id="entities_path" value="{{ entities_path|e }}"/>

      <label>Relations JSON path</label>
      <input id="relations_path" value="{{ relations_path|e }}"/>

      <div class="row">
        <div>
          <label>Use LLM</label>
          <select id="use_llm">
            <option value="true">true</option>
            <option value="false">false</option>
          </select>
        </div>
        <div>
          <label>OpenAI base URL</label>
          <input id="api_base" value="{{ api_base|e }}"/>
        </div>
      </div>

      <div class="row">
        <div>
          <label>API key</label>
          <input id="api_key" value="{{ api_key|e }}" type="password"/>
        </div>
        <div>
          <label>Model</label>
          <input id="api_model" value="{{ api_model|e }}"/>
        </div>
      </div>

      <div class="row">
        <div>
          <label>Retrieval depth</label>
          <input id="depth" value="2" type="number" min="1" max="6"/>
        </div>
        <div>
          <label>Max seeds</label>
          <input id="max_seeds" value="30" type="number" min="1" max="500"/>
        </div>
      </div>

      <div class="row">
        <div>
          <label>Max triples</label>
          <input id="max_triples" value="250" type="number" min="10" max="5000"/>
        </div>
        <div>
          <label>Top-K</label>
          <input id="top_k" value="8" type="number" min="1" max="100"/>
        </div>
      </div>

      <label>Query</label>
      <textarea id="query" placeholder="Example: Recommend GPUs with >= 16GB VRAM and TDP <= 250W. Include trace."></textarea>

      <div class="smallbtns">
        <button id="btnRun">Run</button>
        <button class="secondary" id="btnClear">Clear Output</button>
      </div>

      <div id="loadMsg" class="muted" style="margin-top:10px;"></div>
    </div>

    <div class="card">
      <h2>Output</h2>
      <div id="runMeta" class="muted"></div>
      <div id="out"></div>
      <details id="ctxBox">
        <summary>Context pack (LLM evidence input)</summary>
        <pre id="ctx" class="mono" style="white-space:pre-wrap; margin:10px 0 0 0;"></pre>
      </details>
      <details id="llmBox">
        <summary>LLM answer (evidence-only)</summary>
        <pre id="llm" class="mono" style="white-space:pre-wrap; margin:10px 0 0 0;"></pre>
      </details>
      <details id="statsBox">
        <summary>KG summary stats</summary>
        <pre id="stats" class="mono" style="white-space:pre-wrap; margin:10px 0 0 0;"></pre>
      </details>
    </div>
  </div>

<script>
async function postJSON(url, payload) {
  const r = await fetch(url, { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify(payload) });
  const t = await r.text();
  let j = null;
  try { j = JSON.parse(t); } catch (e) { j = { ok:false, error:"Invalid JSON response", raw:t }; }
  return { status:r.status, json:j };
}

function el(id){ return document.getElementById(id); }

function setStats(stats) {
  if (!stats) return;
  el("st_triples").textContent = stats.triple_count ?? "-";
  el("st_entities").textContent = stats.entity_count ?? "-";
  el("st_relations").textContent = stats.relation_count ?? "-";
  el("st_loaded").textContent = stats.triple_count !== undefined ? "yes" : "no";
  el("stats").textContent = JSON.stringify(stats, null, 2);
}

function esc(s) {
  return (s ?? "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}

function evidenceLineHTML(ev) {
  const left = `<span class="mono">${esc(ev.s_label)}</span> <span class="muted mono">(${esc(ev.s_id)})</span>`;
  const mid = `<span class="mono">${esc(ev.p)}</span>`;
  const right = `<span class="mono">${esc(ev.o_label)}</span> <span class="muted mono">(${esc(ev.o_id)})</span>`;
  const kind = `<span class="pill">${esc(ev.kind)}</span>`;
  const src = ev.source_url ? `<div class="muted">source: <span class="mono">${esc(ev.source_url)}</span></div>` : "";
  return `<div class="mono">• ${left} <span class="muted">|</span> ${mid} <span class="muted">|</span> ${right} ${kind}</div>${src}`;
}

function renderResults(res) {
  const out = el("out");
  out.innerHTML = "";
  const meta = [];
  meta.push(`Retrieved nodes: ${res.retrieved_nodes}`);
  meta.push(`Evidence triples: ${res.evidence_triples}`);
  meta.push(`Evidence relations(JSON): ${res.evidence_relations}`);
  meta.push(`Elapsed: ${res.elapsed_s.toFixed(2)}s`);
  el("runMeta").textContent = meta.join("  |  ");

  const items = res.items || [];
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    const reasons = (it.reasons || []).map(r => `<div class="mono">- ${esc(r)}</div>`).join("");
    const urls = (it.source_urls || []).map(u => `<div class="mono">- ${esc(u)}</div>`).join("");
    const ev = (it.evidence_pretty || []).map(e => evidenceLineHTML(e)).join("<div style='height:8px;'></div>");
    const block = `
      <div class="item">
        <h3>${i+1}. ${esc(it.item_name)} <span class="muted mono">(${esc(it.item_id)})</span></h3>
        <div class="kv"><div>Score</div><div class="mono">${esc(it.score.toFixed(3))}</div></div>
        <div class="kv"><div>Reasons</div><div>${reasons || "<span class='muted'>No reasons</span>"}</div></div>
        <details>
          <summary>Trace (evidence with node labels)</summary>
          <div style="margin-top:10px;">${ev || "<span class='muted'>No trace lines</span>"}</div>
        </details>
        <details>
          <summary>Source URLs</summary>
          <div style="margin-top:10px;">${urls || "<span class='muted'>No URLs</span>"}</div>
        </details>
      </div>
    `;
    out.insertAdjacentHTML("beforeend", block);
  }

  el("ctx").textContent = res.context_pack || "";
  el("llm").textContent = res.llm_answer || "";
  el("ctxBox").open = false;
  el("llmBox").open = false;
}

function payloadFromUI() {
  return {
    ttl_path: el("ttl_path").value.trim(),
    entities_json_path: el("entities_path").value.trim(),
    relations_json_path: el("relations_path").value.trim(),
    query: el("query").value.trim(),
    use_llm: el("use_llm").value === "true",
    api_base: el("api_base").value.trim(),
    api_key: el("api_key").value.trim(),
    api_model: el("api_model").value.trim(),
    depth: parseInt(el("depth").value || "2"),
    max_seeds: parseInt(el("max_seeds").value || "30"),
    max_triples: parseInt(el("max_triples").value || "250"),
    top_k: parseInt(el("top_k").value || "8")
  };
}

async function refreshStats() {
  const r = await fetch("/api/stats");
  const t = await r.text();
  let j = null;
  try { j = JSON.parse(t); } catch (e) { j = { ok:false, error:"Invalid JSON", raw:t }; }
  if (j.ok) {
    setStats(j.stats);
    el("loadMsg").innerHTML = `<span class="ok">Stats updated.</span>`;
  } else {
    el("loadMsg").innerHTML = `<span class="err">${esc(j.error || "Stats error")}</span>`;
  }
}

async function reloadKG() {
  const p = payloadFromUI();
  const { json } = await postJSON("/api/reload", {
    ttl_path: p.ttl_path,
    entities_json_path: p.entities_json_path,
    relations_json_path: p.relations_json_path
  });
  if (json.ok) {
    setStats(json.stats);
    el("loadMsg").innerHTML = `<span class="ok">KG reloaded.</span>`;
  } else {
    el("loadMsg").innerHTML = `<span class="err">${esc(json.error || "Reload error")}</span>`;
  }
}

async function run() {
  const p = payloadFromUI();
  if (!p.query) {
    el("loadMsg").innerHTML = `<span class="err">Query is required.</span>`;
    return;
  }
  el("loadMsg").innerHTML = `<span class="muted">Running...</span>`;
  const { json } = await postJSON("/api/run", p);
  if (json.ok) {
    renderResults(json.result);
    el("loadMsg").innerHTML = `<span class="ok">Done.</span>`;
  } else {
    el("loadMsg").innerHTML = `<span class="err">${esc(json.error || "Run error")}</span>`;
    if (json.stats) setStats(json.stats);
  }
}

el("btnRun").addEventListener("click", run);
el("btnClear").addEventListener("click", () => {
  el("out").innerHTML = "";
  el("ctx").textContent = "";
  el("llm").textContent = "";
  el("runMeta").textContent = "";
  el("loadMsg").textContent = "";
});
el("btnReload").addEventListener("click", reloadKG);
el("btnStats").addEventListener("click", refreshStats);

(async function init() {
  await refreshStats();
})();
</script>
</body>
</html>
"""


def _sanitize_paths(ttl_path: str, entities_path: str, relations_path: str) -> Tuple[bool, str]:
    for p in [ttl_path, entities_path, relations_path]:
        if not p:
            return False, "All KG file paths are required."
        if not _is_pathlike(p):
            return False, "Paths must be local filesystem paths."
    return True, ""


@app.get("/")
def index():
    return render_template_string(
        HTML,
        ttl_path=DEFAULT_TTL_PATH,
        entities_path=DEFAULT_ENTITIES_JSON_PATH,
        relations_path=DEFAULT_RELATIONS_JSON_PATH,
        api_base=DEFAULT_OPENAI_BASE_URL,
        api_key=DEFAULT_OPENAI_API_KEY,
        api_model=DEFAULT_OPENAI_MODEL,
    )


@app.get("/api/stats")
def api_stats():
    if GLOBAL_STATS:
        return jsonify({"ok": True, "stats": GLOBAL_STATS})
    ok, err, stats = load_kb(DEFAULT_TTL_PATH, DEFAULT_ENTITIES_JSON_PATH, DEFAULT_RELATIONS_JSON_PATH)
    if ok:
        return jsonify({"ok": True, "stats": stats})
    return jsonify({"ok": False, "error": err})


@app.post("/api/reload")
def api_reload():
    payload = request.get_json(force=True, silent=True) or {}
    ttl_path = _safe_str(payload.get("ttl_path", DEFAULT_TTL_PATH)).strip()
    entities_path = _safe_str(payload.get("entities_json_path", DEFAULT_ENTITIES_JSON_PATH)).strip()
    relations_path = _safe_str(payload.get("relations_json_path", DEFAULT_RELATIONS_JSON_PATH)).strip()
    okp, perr = _sanitize_paths(ttl_path, entities_path, relations_path)
    if not okp:
        return jsonify({"ok": False, "error": perr})

    ok, err, stats = load_kb(ttl_path, entities_path, relations_path)
    if ok:
        return jsonify({"ok": True, "stats": stats})
    return jsonify({"ok": False, "error": err})


@app.post("/api/run")
def api_run():
    payload = request.get_json(force=True, silent=True) or {}

    ttl_path = _safe_str(payload.get("ttl_path", DEFAULT_TTL_PATH)).strip()
    entities_path = _safe_str(payload.get("entities_json_path", DEFAULT_ENTITIES_JSON_PATH)).strip()
    relations_path = _safe_str(payload.get("relations_json_path", DEFAULT_RELATIONS_JSON_PATH)).strip()
    query = _safe_str(payload.get("query", "")).strip()

    use_llm = bool(payload.get("use_llm", True))
    api_base = _safe_str(payload.get("api_base", DEFAULT_OPENAI_BASE_URL)).strip()
    api_key = _safe_str(payload.get("api_key", DEFAULT_OPENAI_API_KEY)).strip()
    api_model = _safe_str(payload.get("api_model", DEFAULT_OPENAI_MODEL)).strip()

    depth = int(payload.get("depth", 2) or 2)
    max_seeds = int(payload.get("max_seeds", 30) or 30)
    max_triples = int(payload.get("max_triples", 250) or 250)
    top_k = int(payload.get("top_k", 8) or 8)

    if not query:
        return jsonify({"ok": False, "error": "Query is required.", "stats": GLOBAL_STATS or {}})

    okp, perr = _sanitize_paths(ttl_path, entities_path, relations_path)
    if not okp:
        return jsonify({"ok": False, "error": perr, "stats": GLOBAL_STATS or {}})

    if (not GLOBAL_STATS) or (GLOBAL_KB.loaded_ttl_path != ttl_path or GLOBAL_KB.loaded_entities_json_path != entities_path or GLOBAL_KB.loaded_relations_json_path != relations_path):
        ok, err, stats = load_kb(ttl_path, entities_path, relations_path)
        if not ok:
            return jsonify({"ok": False, "error": err, "stats": stats or GLOBAL_STATS or {}})

    t0 = time.time()
    ret = retrieve_subgraph(GLOBAL_KB, query, max_seeds=max_seeds, depth=depth, max_triples=max_triples)
    ranked = rank_technologies(GLOBAL_KB, ret, query, top_k=top_k)
    llm_answer = ""

    if use_llm:
        if api_base and api_key and api_model:
            client = LLMClient(api_base, api_key, api_model)
            sys, user = build_llm_prompt(query, ranked, ret.context_pack_text)
            out = client.chat(sys, user, temperature=0.0)
            if validate_llm_output(ret.context_pack_text, out):
                llm_answer = out
            else:
                llm_answer = "LLM_OUTPUT_REJECTED"
        else:
            llm_answer = "LLM_NOT_CONFIGURED"

    elapsed = time.time() - t0

    items_out: List[Dict[str, Any]] = []
    for it in ranked:
        evs = []
        for ev in it.evidence_pretty:
            evs.append({
                "raw": ev.raw,
                "s_id": ev.s_id,
                "p": ev.p,
                "o_id": ev.o_id,
                "s_label": ev.s_label,
                "o_label": ev.o_label,
                "kind": ev.kind,
                "source_url": ev.source_url,
            })
        items_out.append({
            "item_id": it.item_id,
            "item_name": it.item_name,
            "score": it.score,
            "reasons": it.reasons,
            "evidence_pretty": evs,
            "source_urls": it.source_urls,
        })

    result = {
        "retrieved_nodes": len(ret.candidate_nodes),
        "evidence_triples": len(ret.evidence_triples),
        "evidence_relations": len(ret.evidence_relations),
        "elapsed_s": elapsed,
        "items": items_out,
        "context_pack": ret.context_pack_text,
        "llm_answer": llm_answer,
        "constraints": parse_query_constraints(query),
    }

    return jsonify({"ok": True, "result": result, "stats": GLOBAL_STATS or {}})


if __name__ == "__main__":
    load_kb(DEFAULT_TTL_PATH, DEFAULT_ENTITIES_JSON_PATH, DEFAULT_RELATIONS_JSON_PATH)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=False)
