import json
import os
import re
import threading
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
        line = f"{s} {p} {o}"
        trace.setdefault(s, []).append(line)
        trace.setdefault(o, []).append(line)
    for (s, p, o, st, su) in rels:
        line = f"{s} {p} {o}"
        trace.setdefault(s, []).append(line)
        trace.setdefault(o, []).append(line)

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
        for gid in gpu_candidates[:80]:
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
            if s == cid or o == cid:
                if su:
                    urls.add(su)

        trace_lines = retrieval.trace.get(cid, [])
        ev_lines.extend(trace_lines)
        if not ev_lines:
            ev_lines.append(cid)

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
