import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS


DEFAULT_TTL_PATH = "/mnt/data/e27afcf6-5f87-40ad-b2e8-01bc625f6f0c.ttl"
DEFAULT_ENTITIES_JSON_PATH = "/mnt/data/60033a0d-87f9-4756-8862-029bf897c1e4.json"
DEFAULT_RELATIONS_JSON_PATH = "/mnt/data/a299efd0-2cac-47dc-a02c-d0c058df0cbb.json"


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


def _pretty_local_name(s: str) -> str:
    if "#" in s:
        return s.split("#")[-1]
    if "/" in s:
        return s.rstrip("/").split("/")[-1]
    return s


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _extract_numbers(text: str) -> Set[str]:
    nums = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text))
    return nums


def _tokenize(text: str) -> List[str]:
    t = re.sub(r"[^a-zA-Z0-9:_\-]+", " ", text.lower()).strip()
    return [x for x in t.split() if x]


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
    m = re.search(r"(\d+(?:\.\d+)?)\s*w", q, re.IGNORECASE)
    if "max_tdp_w" not in out and m and re.search(r"(<=|<|at\s*most|max)", q, re.IGNORECASE):
        out["max_tdp_w"] = float(m.group(1))
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
            self.relations.append(r)
        self.loaded_relations_json_path = path

    def build_index(self) -> None:
        self.node_text_index = {}
        self.node_tokens = {}
        for s, p, o in self.g:
            sid = _safe_str(s)
            oid = _safe_str(o)
            if sid not in self.node_text_index:
                self.node_text_index[sid] = ""
            if oid.startswith("http://") or oid.startswith("https://"):
                self.node_text_index[oid] = self.node_text_index.get(oid, "")
        for s in set([_safe_str(x) for x in self.g.subjects()] + [_safe_str(x) for x in self.g.objects()]):
            label = self.get_label_for_node_str(s)
            self.node_text_index[s] = (self.node_text_index.get(s, "") + " " + label).strip()
        for eid, e in self.entities.items():
            self.node_text_index[eid] = (self.node_text_index.get(eid, "") + " " + e.name + " " + e.source_text + " " + e.etype + " " + e.category).strip()
        for nid, text in self.node_text_index.items():
            self.node_tokens[nid] = set(_tokenize(text))

    def get_label_for_node_str(self, node_str: str) -> str:
        if node_str in self.entities:
            return self.entities[node_str].name
        try:
            node = URIRef(node_str) if (node_str.startswith("http://") or node_str.startswith("https://")) else URIRef(node_str)
            for lbl in self.g.objects(node, RDFS.label):
                if isinstance(lbl, Literal):
                    return _safe_str(lbl)
        except Exception:
            pass
        return _pretty_local_name(node_str)

    def get_triples_for_node(self, node_str: str) -> List[Tuple[str, str, str]]:
        triples: List[Tuple[str, str, str]] = []
        try:
            node = URIRef(node_str) if (node_str.startswith("http://") or node_str.startswith("https://")) else URIRef(node_str)
            for s, p, o in self.g.triples((node, None, None)):
                triples.append((_safe_str(s), _safe_str(p), _safe_str(o)))
            for s, p, o in self.g.triples((None, None, node)):
                triples.append((_safe_str(s), _safe_str(p), _safe_str(o)))
        except Exception:
            pass
        return triples

    def neighbors(self, node_str: str) -> Set[str]:
        out: Set[str] = set()
        try:
            node = URIRef(node_str) if (node_str.startswith("http://") or node_str.startswith("https://")) else URIRef(node_str)
            for s, p, o in self.g.triples((node, None, None)):
                out.add(_safe_str(o))
            for s, p, o in self.g.triples((None, None, node)):
                out.add(_safe_str(s))
        except Exception:
            pass
        for r in self.relations:
            if r.source_id == node_str:
                out.add(r.target_id)
            if r.target_id == node_str:
                out.add(r.source_id)
        return out

    def relation_triples_for_pair(self, a: str, b: str) -> List[Tuple[str, str, str, str, str]]:
        out: List[Tuple[str, str, str, str, str]] = []
        for r in self.relations:
            if r.source_id == a and r.target_id == b:
                out.append((r.source_id, r.predicate, r.target_id, r.source_text, r.source_url))
            if r.source_id == b and r.target_id == a:
                out.append((r.source_id, r.predicate, r.target_id, r.source_text, r.source_url))
        return out


@dataclass
class RetrievalResult:
    candidate_nodes: List[str]
    evidence_triples: List[Tuple[str, str, str]]
    evidence_relations: List[Tuple[str, str, str, str, str]]
    context_pack_text: str
    trace: Dict[str, List[str]]


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
    trace: Dict[str, List[str]] = {}
    for (s, p, o) in triples:
        trace.setdefault(s, []).append(f"{s} {p} {o}")
        trace.setdefault(o, []).append(f"{s} {p} {o}")
    for (s, p, o, st, su) in rels:
        trace.setdefault(s, []).append(f"{s} {p} {o}")
        trace.setdefault(o, []).append(f"{s} {p} {o}")
    context_lines: List[str] = []
    context_lines.append("EVIDENCE_TRIPLES:")
    for (s, p, o) in triples:
        context_lines.append(f"- {kb.get_label_for_node_str(s)} | { _pretty_local_name(p) } | {kb.get_label_for_node_str(o)}")
    if rels:
        context_lines.append("EVIDENCE_RELATIONS_JSON:")
        for (s, p, o, st, su) in rels:
            context_lines.append(f"- {kb.get_label_for_node_str(s)} | {p} | {kb.get_label_for_node_str(o)} | src: {su}")
    attrs_lines: List[str] = []
    gpu_candidates = [n for n in nodes if (n in kb.entities and kb.entities[n].etype.lower() == "gpu")]
    if gpu_candidates:
        attrs_lines.append("GPU_ATTRIBUTES_FROM_JSON:")
        for gid in gpu_candidates[:60]:
            e = kb.entities.get(gid)
            if not e:
                continue
            specs = kb.gpu_specs.get(gid, {})
            if specs:
                attrs_lines.append(f"- {e.name} ({gid}) specs: {json.dumps(specs, ensure_ascii=False)}")
            else:
                attrs_lines.append(f"- {e.name} ({gid}) specs: {{}}")
    context_pack = "\n".join(context_lines + attrs_lines)
    return RetrievalResult(
        candidate_nodes=nodes,
        evidence_triples=triples,
        evidence_relations=rels,
        context_pack_text=context_pack,
        trace=trace,
    )


@dataclass
class RankedItem:
    item_id: str
    item_name: str
    score: float
    reasons: List[str]
    evidence_lines: List[str]
    source_urls: List[str]


def rank_technologies(kb: KnowledgeBase, retrieval: RetrievalResult, query: str, top_k: int = 10) -> List[RankedItem]:
    constraints = parse_query_constraints(query)
    candidates: List[str] = []
    for n in retrieval.candidate_nodes:
        if n in kb.entities and kb.entities[n].etype.lower() == "gpu":
            candidates.append(n)
    seen: Set[str] = set()
    candidates_unique = []
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
        ev_lines: List[str] = []

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

        for (s, p, o, st, su) in retrieval.evidence_relations:
            if s == cid:
                urls.add(su)
            if o == cid:
                urls.add(su)

        trace_lines = retrieval.trace.get(cid, [])
        for tl in trace_lines:
            ev_lines.append(tl)
        if not ev_lines:
            ev_lines.append(cid)

        if constraints.get("manufacturer"):
            mfg = constraints["manufacturer"]
            mfg_match = False
            for (s, pred, o, st, su) in retrieval.evidence_relations:
                if o == cid and pred.lower() == "produces" and s in kb.entities:
                    if kb.entities[s].name.strip().upper() == mfg:
                        mfg_match = True
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
                        urls.add(su)
            if arch_match:
                score += 1.2
                reasons.append("Architecture matches query hint")
            else:
                reasons.append("Architecture not confirmed as query hint in retrieved evidence")

        scored.append(
            RankedItem(
                item_id=cid,
                item_name=e.name,
                score=score,
                reasons=reasons,
                evidence_lines=ev_lines[:25],
                source_urls=sorted(list(urls))[:10],
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
        return _safe_str(data["choices"][0]["message"]["content"])


def build_llm_prompt(query: str, ranked: List[RankedItem], context_pack: str) -> Tuple[str, str]:
    sys = (
        "You are a constrained assistant. You must ONLY use the provided EVIDENCE to produce output.\n"
        "Rules:\n"
        "1) Do not introduce any fact, number, comparison, or attribute not explicitly present in EVIDENCE.\n"
        "2) If EVIDENCE is insufficient for any part, write 'INSUFFICIENT_EVIDENCE'.\n"
        "3) You must cite trace references using item_id and evidence triple lines exactly.\n"
        "4) Output must be deterministic and grounded.\n"
        "5) Do not mention policies. Do not add creativity.\n"
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


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("KG-Grounded RAG Recommender (Traceable)")
        self.geometry("1180x760")

        self.kb = KnowledgeBase()
        self.retrieval: Optional[RetrievalResult] = None
        self.ranked: List[RankedItem] = []
        self.llm_answer: str = ""

        self.ttl_path = tk.StringVar(value=DEFAULT_TTL_PATH)
        self.entities_json_path = tk.StringVar(value=DEFAULT_ENTITIES_JSON_PATH)
        self.relations_json_path = tk.StringVar(value=DEFAULT_RELATIONS_JSON_PATH)

        self.api_base = tk.StringVar(value=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        self.api_key = tk.StringVar(value=os.environ.get("OPENAI_API_KEY", ""))
        self.api_model = tk.StringVar(value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

        self.use_llm = tk.BooleanVar(value=True)
        self.depth = tk.IntVar(value=2)
        self.max_triples = tk.IntVar(value=250)
        self.top_k = tk.IntVar(value=8)

        self._build_ui()

    def _build_ui(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        frame_load = ttk.Frame(nb)
        frame_query = ttk.Frame(nb)
        frame_results = ttk.Frame(nb)
        frame_evidence = ttk.Frame(nb)

        nb.add(frame_load, text="1) Load KG/KB")
        nb.add(frame_query, text="2) Query + Retrieve")
        nb.add(frame_results, text="3) Ranked Output")
        nb.add(frame_evidence, text="4) Evidence + Trace")

        self._build_load_tab(frame_load)
        self._build_query_tab(frame_query)
        self._build_results_tab(frame_results)
        self._build_evidence_tab(frame_evidence)

    def _row(self, parent: tk.Widget, r: int) -> ttk.Frame:
        f = ttk.Frame(parent)
        f.grid(row=r, column=0, sticky="ew", padx=12, pady=6)
        f.columnconfigure(1, weight=1)
        return f

    def _build_load_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        r0 = self._row(parent, 0)
        ttk.Label(r0, text="TTL Path:").grid(row=0, column=0, sticky="w")
        ttk.Entry(r0, textvariable=self.ttl_path).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(r0, text="Browse", command=self._browse_ttl).grid(row=0, column=2, padx=6)

        r1 = self._row(parent, 1)
        ttk.Label(r1, text="Entities JSON Path:").grid(row=0, column=0, sticky="w")
        ttk.Entry(r1, textvariable=self.entities_json_path).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(r1, text="Browse", command=self._browse_entities).grid(row=0, column=2, padx=6)

        r2 = self._row(parent, 2)
        ttk.Label(r2, text="Relations JSON Path:").grid(row=0, column=0, sticky="w")
        ttk.Entry(r2, textvariable=self.relations_json_path).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(r2, text="Browse", command=self._browse_relations).grid(row=0, column=2, padx=6)

        r3 = self._row(parent, 3)
        ttk.Separator(r3, orient="horizontal").grid(row=0, column=0, columnspan=3, sticky="ew", pady=6)

        r4 = self._row(parent, 4)
        ttk.Label(r4, text="LLM Base URL:").grid(row=0, column=0, sticky="w")
        ttk.Entry(r4, textvariable=self.api_base).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Label(r4, text="Model:").grid(row=0, column=2, sticky="w", padx=6)
        ttk.Entry(r4, textvariable=self.api_model, width=24).grid(row=0, column=3, sticky="w")

        r5 = self._row(parent, 5)
        ttk.Label(r5, text="API Key:").grid(row=0, column=0, sticky="w")
        e = ttk.Entry(r5, textvariable=self.api_key, show="•")
        e.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Checkbutton(r5, text="Use LLM (still evidence-only)", variable=self.use_llm).grid(row=0, column=2, padx=8, sticky="w")

        r6 = self._row(parent, 6)
        ttk.Button(r6, text="Load + Build Index", command=self._load_all).grid(row=0, column=0, sticky="w")
        self.load_status = ttk.Label(r6, text="Not loaded")
        self.load_status.grid(row=0, column=1, sticky="w", padx=10)

        r7 = self._row(parent, 7)
        self.kg_stats = tk.Text(parent, height=18, wrap="word")
        self.kg_stats.grid(row=7, column=0, sticky="nsew", padx=12, pady=10)
        parent.rowconfigure(7, weight=1)

    def _build_query_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        r0 = self._row(parent, 0)
        ttk.Label(r0, text="Query:").grid(row=0, column=0, sticky="nw")
        self.query_text = tk.Text(r0, height=5, wrap="word")
        self.query_text.grid(row=0, column=1, sticky="ew", padx=8)
        r0.rowconfigure(0, weight=1)

        r1 = self._row(parent, 1)
        ttk.Label(r1, text="Retrieval depth:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(r1, from_=1, to=4, textvariable=self.depth, width=5).grid(row=0, column=1, sticky="w")
        ttk.Label(r1, text="Max triples:").grid(row=0, column=2, sticky="w", padx=10)
        ttk.Spinbox(r1, from_=50, to=1000, increment=50, textvariable=self.max_triples, width=7).grid(row=0, column=3, sticky="w")
        ttk.Label(r1, text="Top-K:").grid(row=0, column=4, sticky="w", padx=10)
        ttk.Spinbox(r1, from_=3, to=20, textvariable=self.top_k, width=5).grid(row=0, column=5, sticky="w")

        r2 = self._row(parent, 2)
        ttk.Button(r2, text="Retrieve + Rank (+ Optional LLM)", command=self._run_pipeline).grid(row=0, column=0, sticky="w")
        self.run_status = ttk.Label(r2, text="")
        self.run_status.grid(row=0, column=1, sticky="w", padx=10)

        self.query_log = tk.Text(parent, height=22, wrap="word")
        self.query_log.grid(row=3, column=0, sticky="nsew", padx=12, pady=10)
        parent.rowconfigure(3, weight=1)

    def _build_results_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        self.results_text = tk.Text(parent, wrap="word")
        self.results_text.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_evidence_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        self.evidence_text = tk.Text(parent, wrap="word")
        self.evidence_text.pack(fill="both", expand=True, padx=12, pady=12)

    def _browse_ttl(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Turtle", "*.ttl"), ("All files", "*.*")])
        if p:
            self.ttl_path.set(p)

    def _browse_entities(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if p:
            self.entities_json_path.set(p)

    def _browse_relations(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if p:
            self.relations_json_path.set(p)

    def _load_all(self) -> None:
        try:
            ttl = self.ttl_path.get().strip()
            ej = self.entities_json_path.get().strip()
            rj = self.relations_json_path.get().strip()

            if ttl:
                self.kb.load_ttl(ttl)
            if ej and os.path.exists(ej):
                self.kb.load_entities_json(ej)
            if rj and os.path.exists(rj):
                self.kb.load_relations_json(rj)

            self.kb.build_index()

            n_triples = len(self.kb.g)
            n_entities = len(self.kb.entities)
            n_rels = len(self.kb.relations)

            self.load_status.config(text=f"Loaded: TTL triples={n_triples}, entities={n_entities}, relations={n_rels}")
            self.kg_stats.delete("1.0", "end")
            self.kg_stats.insert("end", f"TTL: {self.kb.loaded_ttl_path}\n")
            self.kg_stats.insert("end", f"Entities JSON: {self.kb.loaded_entities_json_path}\n")
            self.kg_stats.insert("end", f"Relations JSON: {self.kb.loaded_relations_json_path}\n\n")
            self.kg_stats.insert("end", f"Graph triple count: {n_triples}\n")
            self.kg_stats.insert("end", f"Entity count: {n_entities}\n")
            self.kg_stats.insert("end", f"Relation count: {n_rels}\n\n")
            if n_entities:
                types: Dict[str, int] = {}
                for e in self.kb.entities.values():
                    types[e.etype] = types.get(e.etype, 0) + 1
                self.kg_stats.insert("end", "Entity types:\n")
                for k, v in sorted(types.items(), key=lambda x: (-x[1], x[0])):
                    self.kg_stats.insert("end", f"- {k}: {v}\n")

        except Exception as ex:
            messagebox.showerror("Load error", _safe_str(ex))

    def _run_pipeline(self) -> None:
        q = self.query_text.get("1.0", "end").strip()
        if not q:
            messagebox.showwarning("Missing query", "Please enter a query.")
            return
        if len(self.kb.g) == 0 and not self.kb.entities:
            messagebox.showwarning("Not loaded", "Load KG/KB first (TTL and/or JSON).")
            return

        self.run_status.config(text="Running...")
        self.query_log.delete("1.0", "end")
        self.results_text.delete("1.0", "end")
        self.evidence_text.delete("1.0", "end")
        self.llm_answer = ""
        self.ranked = []
        self.retrieval = None

        def worker() -> None:
            try:
                t0 = time.time()
                ret = retrieve_subgraph(self.kb, q, depth=int(self.depth.get()), max_triples=int(self.max_triples.get()))
                ranked = rank_technologies(self.kb, ret, q, top_k=int(self.top_k.get()))
                llm_out = ""
                if bool(self.use_llm.get()):
                    base = self.api_base.get().strip()
                    key = self.api_key.get().strip()
                    model = self.api_model.get().strip()
                    if base and key and model:
                        client = LLMClient(base, key, model)
                        sys, user = build_llm_prompt(q, ranked, ret.context_pack_text)
                        out = client.chat(sys, user, temperature=0.0)
                        if validate_llm_output(ret.context_pack_text, out):
                            llm_out = out
                        else:
                            llm_out = "LLM_OUTPUT_REJECTED"
                    else:
                        llm_out = "LLM_NOT_CONFIGURED"
                t1 = time.time()

                self.retrieval = ret
                self.ranked = ranked
                self.llm_answer = llm_out

                self.after(0, lambda: self._render(q, t1 - t0))
            except Exception as ex:
                self.after(0, lambda: messagebox.showerror("Run error", _safe_str(ex)))
            finally:
                self.after(0, lambda: self.run_status.config(text="Done"))

        threading.Thread(target=worker, daemon=True).start()

    def _render(self, q: str, elapsed: float) -> None:
        ret = self.retrieval
        ranked = self.ranked
        if not ret:
            return

        self.query_log.insert("end", f"Query:\n{q}\n\n")
        self.query_log.insert("end", f"Retrieved nodes: {len(ret.candidate_nodes)}\n")
        self.query_log.insert("end", f"Evidence triples: {len(ret.evidence_triples)}\n")
        self.query_log.insert("end", f"Evidence relations (JSON): {len(ret.evidence_relations)}\n")
        self.query_log.insert("end", f"Elapsed: {elapsed:.2f}s\n\n")
        self.query_log.insert("end", "Parsed constraints:\n")
        self.query_log.insert("end", json.dumps(parse_query_constraints(q), indent=2, ensure_ascii=False))
        self.query_log.insert("end", "\n")

        self.results_text.insert("end", "RANKED_LIST:\n\n")
        for i, item in enumerate(ranked, start=1):
            self.results_text.insert("end", f"{i}. {item.item_name} ({item.item_id})\n")
            self.results_text.insert("end", f"   Score: {item.score:.3f}\n")
            self.results_text.insert("end", "   Reasons:\n")
            for r in item.reasons:
                self.results_text.insert("end", f"   - {r}\n")
            self.results_text.insert("end", "   Trace (triples/relations lines):\n")
            for ev in item.evidence_lines:
                self.results_text.insert("end", f"   - {ev}\n")
            if item.source_urls:
                self.results_text.insert("end", "   Source URLs:\n")
                for u in item.source_urls:
                    self.results_text.insert("end", f"   - {u}\n")
            self.results_text.insert("end", "\n")

        if self.llm_answer:
            self.results_text.insert("end", "\nLLM_ANSWER (EVIDENCE-ONLY):\n\n")
            self.results_text.insert("end", self.llm_answer.strip() + "\n")

        self.evidence_text.insert("end", "CONTEXT_PACK (WHAT THE LLM IS ALLOWED TO SEE):\n\n")
        self.evidence_text.insert("end", ret.context_pack_text + "\n")

        self.evidence_text.insert("end", "\n\nTRACE_MAP (node -> evidence lines count):\n\n")
        for k, v in sorted(ret.trace.items(), key=lambda x: (-len(x[1]), x[0]))[:120]:
            self.evidence_text.insert("end", f"- {self.kb.get_label_for_node_str(k)} ({k}): {len(v)}\n")



if __name__ == "__main__":
    app = App()
    app.mainloop()