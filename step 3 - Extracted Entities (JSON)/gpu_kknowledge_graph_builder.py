#!/usr/bin/env python3
"""
Build GPU knowledge graph relations (foreign-key JSON) + TTL from extracted entity lists.

Inputs (JSON files):
- gpu_products.json            (GPU_product_entities list)
- manufacturers.json           (manufacturer_entities list)
- architectures.json           (architecture_entities list)
- memory_types.json            (memory_type_entities list)

Outputs:
- kg_entities.json             (all entities with stable IDs)
- kg_relations.json            (foreign-key relations)
- gpu_kg.ttl                   (TTL knowledge graph)

Relations extracted:
- manufacturer ---produces--> gpu
- gpu ---hasArchitecture--> architecture
- gpu ---hasMemoryType--> memory_type

Notes:
- Manufacturer is inferred from GPU name (Radeon -> AMD, otherwise NVIDIA if RTX/GeForce/RTX PRO, else Unknown).
- Architecture is inferred from GPU name using simple rules:
    RTX 50xx or "Blackwell" -> Blackwell
    RTX 40xx or "Ada" -> Ada Lovelace
    Radeon RX 7000 / 7900 / "RDNA 3" -> RDNA 3
  If none applies, the architecture relation is not added.
- Memory type is extracted from GPU source_text (e.g., "Memory GDDR7").
- All relations keep traceability via source_url.

Run:
  python build_gpu_kg.py \
    --gpus gpu_products.json \
    --manufacturers manufacturers.json \
    --architectures architectures.json \
    --memory-types memory_types.json \
    --out-prefix gpu_kg
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# -----------------------
# Utilities
# -----------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def slugify(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t

def clean_text(s: str) -> str:
    s = (s or "").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------
# Entity indexing
# -----------------------

@dataclass
class Entity:
    id: str
    entity_name: str
    entity_type: str
    category: str
    source_text: str
    source_url: str

def index_entities(entities: List[Dict[str, Any]], prefix: str) -> Tuple[Dict[str, Entity], Dict[str, str]]:
    """
    Returns:
      - id -> Entity
      - entity_name -> id
    """
    by_id: Dict[str, Entity] = {}
    by_name: Dict[str, str] = {}

    for e in entities:
        name = clean_text(e.get("entity_name", ""))
        etype = clean_text(e.get("entity_type", ""))
        cat = clean_text(e.get("category", ""))
        st = clean_text(e.get("source_text", ""))
        su = clean_text(e.get("source_url", ""))

        if not name or not etype:
            continue

        eid = f"{prefix}:{slugify(name)}"
        # Ensure uniqueness in rare collision cases
        if eid in by_id:
            suffix = 2
            while f"{eid}_{suffix}" in by_id:
                suffix += 1
            eid = f"{eid}_{suffix}"

        ent = Entity(
            id=eid,
            entity_name=name,
            entity_type=etype,
            category=cat or "Hardware/GPU",
            source_text=st,
            source_url=su,
        )
        by_id[eid] = ent
        # Prefer first seen for by_name mapping
        by_name.setdefault(name, eid)

    return by_id, by_name


# -----------------------
# Relation extraction
# -----------------------

def infer_manufacturer_from_gpu_name(gpu_name: str) -> Optional[str]:
    n = gpu_name.lower()
    if "radeon" in n:
        return "AMD"
    if "geforce" in n or n.startswith("rtx ") or "rtx pro" in n or "nvidia" in n:
        return "NVIDIA"
    if "intel" in n or "arc" in n:
        return "Intel"
    return None

def infer_architecture_from_gpu(gpu_name: str, source_text: str) -> Optional[str]:
    n = gpu_name.lower()
    st = source_text.lower()

    # Explicit mention in name/source_text wins
    if "blackwell" in n or "blackwell" in st:
        return "Blackwell"
    if "ada" in st or "ada lovelace" in st:
        return "Ada Lovelace"
    if "rdna 3" in st or "rdna3" in st:
        return "RDNA 3"

    # Heuristic by series
    # RTX 50xx -> Blackwell
    if re.search(r"\brtx\s*50\d{2}\b", n) or re.search(r"\b50\s*series\b", st):
        return "Blackwell"
    # RTX 40xx -> Ada Lovelace
    if re.search(r"\brtx\s*40\d{2}\b", n) or "rtx 40" in n:
        return "Ada Lovelace"
    # RX 7900 / RX 7000 -> RDNA 3
    if re.search(r"\brx\s*79\d{2}\b", n) or "rx 7000" in n or "rx 7900" in n:
        return "RDNA 3"

    return None

def extract_memory_type_from_source_text(source_text: str) -> Optional[str]:
    # Your source_text pattern: "... Memory GDDR7, ..."
    m = re.search(r"\bMemory\s+([A-Za-z0-9\-\+]+)\b", source_text)
    if m:
        return m.group(1)
    # fallback: look for common types
    for t in ["GDDR7", "GDDR6X", "GDDR6", "HBM3", "HBM2", "HBM"]:
        if t.lower() in source_text.lower():
            return t
    return None


def build_relations(
    gpu_entities: List[Dict[str, Any]],
    manufacturer_name_to_id: Dict[str, str],
    architecture_name_to_id: Dict[str, str],
    memory_type_name_to_id: Dict[str, str],
    gpu_name_to_id: Dict[str, str],
) -> List[Dict[str, Any]]:
    relations: List[Dict[str, Any]] = []
    rel_id_counter = 1

    def add_relation(src_id: str, predicate: str, dst_id: str, source_url: str, source_text: str) -> None:
        nonlocal rel_id_counter
        relations.append(
            {
                "relation_id": f"rel:{rel_id_counter:06d}",
                "source_id": src_id,
                "predicate": predicate,
                "target_id": dst_id,
                "source_url": source_url,
                "source_text": source_text,
            }
        )
        rel_id_counter += 1

    for g in gpu_entities:
        gpu_name = clean_text(g.get("entity_name", ""))
        gpu_id = gpu_name_to_id.get(gpu_name)
        if not gpu_id:
            continue

        source_url = clean_text(g.get("source_url", ""))
        source_text = clean_text(g.get("source_text", ""))

        # manufacturer ---produces--> gpu
        mfg = infer_manufacturer_from_gpu_name(gpu_name)
        if mfg and mfg in manufacturer_name_to_id:
            add_relation(
                src_id=manufacturer_name_to_id[mfg],
                predicate="produces",
                dst_id=gpu_id,
                source_url=source_url,
                source_text=f"Inferred manufacturer for {gpu_name}: {mfg}.",
            )

        # gpu ---hasArchitecture--> architecture
        arch = infer_architecture_from_gpu(gpu_name, source_text)
        if arch and arch in architecture_name_to_id:
            add_relation(
                src_id=gpu_id,
                predicate="hasArchitecture",
                dst_id=architecture_name_to_id[arch],
                source_url=source_url,
                source_text=f"Inferred architecture for {gpu_name}: {arch}.",
            )

        # gpu ---hasMemoryType--> memory_type
        mem = extract_memory_type_from_source_text(source_text)
        if mem and mem in memory_type_name_to_id:
            add_relation(
                src_id=gpu_id,
                predicate="hasMemoryType",
                dst_id=memory_type_name_to_id[mem],
                source_url=source_url,
                source_text=f"Memory type for {gpu_name}: {mem}.",
            )

    return relations


# -----------------------
# TTL serialization
# -----------------------

def turtle_escape_literal(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return s

def to_ttl(entities_by_id: Dict[str, Entity], relations: List[Dict[str, Any]], base_iri: str) -> str:
    """
    Simple RDF graph:
      - Entities are IRIs: base_iri + local_id
      - Predicates are IRIs: base_iri + predicate
      - Each entity typed with ex:EntityType (e.g., ex:gpu, ex:manufacturer, ...)
      - Includes rdfs:label and ex:sourceUrl
      - Relations include ex:sourceUrl and ex:sourceText as reified blank nodes? (simple approach: add as annotations on predicate triple)
        We'll keep it simple: add separate triples:
           ex:subject ex:predicate ex:object .
           ex:subject ex:relationEvidence "..." .
           ex:subject ex:relationSourceUrl "..." .
        (Not perfect RDF, but simple and acceptable for coursework.)
    """
    lines: List[str] = []
    lines.append(f"@prefix ex: <{base_iri}> .")
    lines.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
    lines.append("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
    lines.append("")

    # Entities
    for eid, ent in entities_by_id.items():
        local = eid.split(":", 1)[1] if ":" in eid else eid
        subj = f"ex:{local}"

        etype_local = slugify(ent.entity_type)
        label = turtle_escape_literal(ent.entity_name)
        srcu = turtle_escape_literal(ent.source_url)

        lines.append(f"{subj} rdf:type ex:{etype_local} ;")
        lines.append(f'  rdfs:label "{label}" ;')
        lines.append(f'  ex:entityType "{turtle_escape_literal(ent.entity_type)}" ;')
        lines.append(f'  ex:category "{turtle_escape_literal(ent.category)}" ;')
        if ent.source_url:
            lines.append(f'  ex:sourceUrl "{srcu}" ;')
        if ent.source_text:
            lines.append(f'  ex:sourceText "{turtle_escape_literal(ent.source_text)}" ;')
        lines.append("  .")
        lines.append("")

    # Relations
    # We'll also attach evidence as separate triples via a relation node:
    # ex:rel_000001 ex:subject ex:... ; ex:predicate "hasMemoryType" ; ex:object ex:... ; ex:sourceUrl "..."; ex:sourceText "..."
    for rel in relations:
        rid = rel["relation_id"]
        rlocal = rid.split(":", 1)[1]
        rnode = f"ex:rel_{rlocal}"

        s_id = rel["source_id"]
        o_id = rel["target_id"]
        pred = rel["predicate"]

        s_local = s_id.split(":", 1)[1] if ":" in s_id else s_id
        o_local = o_id.split(":", 1)[1] if ":" in o_id else o_id

        lines.append(f"{rnode} rdf:type ex:Relation ;")
        lines.append(f"  ex:subject ex:{s_local} ;")
        lines.append(f'  ex:predicate "{turtle_escape_literal(pred)}" ;')
        lines.append(f"  ex:object ex:{o_local} ;")
        if rel.get("source_url"):
            lines.append(f'  ex:sourceUrl "{turtle_escape_literal(rel["source_url"])}" ;')
        if rel.get("source_text"):
            lines.append(f'  ex:sourceText "{turtle_escape_literal(rel["source_text"])}" ;')
        lines.append("  .")
        lines.append("")

        # Also emit the direct triple for graph querying:
        lines.append(f"ex:{s_local} ex:{pred} ex:{o_local} .")
        lines.append("")

    return "\n".join(lines)


# -----------------------
# Main
# -----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True, help="Path to GPU_product_entities JSON file (a list)")
    ap.add_argument("--manufacturers", required=True, help="Path to manufacturer_entities JSON file (a list)")
    ap.add_argument("--architectures", required=True, help="Path to architecture_entities JSON file (a list)")
    ap.add_argument("--memory-types", required=True, help="Path to memory_type_entities JSON file (a list)")
    ap.add_argument("--out-prefix", default="gpu_kg", help="Prefix for output files")
    ap.add_argument("--base-iri", default="http://example.org/gpu#", help="Base IRI for TTL output (end with # or /)")
    args = ap.parse_args()

    gpu_list = load_json(args.gpus)
    manufacturer_list = load_json(args.manufacturers)
    architecture_list = load_json(args.architectures)
    memory_type_list = load_json(args.memory_types)

    # Index entities and assign IDs
    gpu_by_id, gpu_name_to_id = index_entities(gpu_list, prefix="gpu")
    mfg_by_id, mfg_name_to_id = index_entities(manufacturer_list, prefix="mfg")
    arch_by_id, arch_name_to_id = index_entities(architecture_list, prefix="arch")
    mem_by_id, mem_name_to_id = index_entities(memory_type_list, prefix="mem")

    # Merge all entities into one store
    all_entities_by_id: Dict[str, Entity] = {}
    all_entities_by_id.update(gpu_by_id)
    all_entities_by_id.update(mfg_by_id)
    all_entities_by_id.update(arch_by_id)
    all_entities_by_id.update(mem_by_id)

    # Build relations
    relations = build_relations(
        gpu_entities=gpu_list,
        manufacturer_name_to_id=mfg_name_to_id,
        architecture_name_to_id=arch_name_to_id,
        memory_type_name_to_id=mem_name_to_id,
        gpu_name_to_id=gpu_name_to_id,
    )

    # Output: entities + relations JSON
    entities_out = [
        {
            "id": e.id,
            "entity_name": e.entity_name,
            "entity_type": e.entity_type,
            "category": e.category,
            "source_text": e.source_text,
            "source_url": e.source_url,
        }
        for e in all_entities_by_id.values()
    ]

    out_entities_path = f"{args.out_prefix}_entities.json"
    out_relations_path = f"{args.out_prefix}_relations.json"
    out_ttl_path = f"{args.out_prefix}.ttl"

    save_json(out_entities_path, entities_out)
    save_json(out_relations_path, relations)

    # TTL
    ttl = to_ttl(all_entities_by_id, relations, base_iri=args.base_iri)
    with open(out_ttl_path, "w", encoding="utf-8") as f:
        f.write(ttl)

    print(f"[OK] Wrote entities:  {out_entities_path} ({len(entities_out)})")
    print(f"[OK] Wrote relations: {out_relations_path} ({len(relations)})")
    print(f"[OK] Wrote TTL:       {out_ttl_path}")


if __name__ == "__main__":
    main()
