#!/usr/bin/env python3
"""
TTL Knowledge Graph Visualizer (HTML-only)

What’s improved vs your current version:
✅ HTML output only (no PNG, no matplotlib)
✅ Node coloring by entity type (gpu / manufacturer / architecture / memory_type / Relation / other)
✅ Better labels: uses rdfs:label when available; otherwise pretty local-name
✅ Hierarchical layout option (manufacturer → gpu → architecture/memory_type) for cleaner structure
✅ Physics tuned for nicer spacing + fewer overlaps
✅ Better UX: search box, filter controls, hover tooltips, smooth edges, arrowheads
✅ Hides clutter by default: relation nodes (ex:rel_*) and metadata predicates

Dependencies (inside your venv):
  pip install rdflib pyvis

Run:
  python visualize_ttl.py --ttl gpu_kg.ttl --html gpu_kg_pretty.html
Optional:
  python visualize_ttl.py --ttl gpu_kg.ttl --html gpu_kg_pretty.html --layout hierarchical
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS
from pyvis.network import Network


# -----------------------------
# Helpers
# -----------------------------

def clean_text(s: str) -> str:
    s = (s or "").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def local_name(uri: str) -> str:
    if "#" in uri:
        return uri.split("#")[-1]
    if "/" in uri:
        return uri.rstrip("/").split("/")[-1]
    return uri

def pretty_local(local: str) -> str:
    # ex:geforce_rtx_4090 -> GeForce Rtx 4090 (best-effort)
    t = local.replace("_", " ").strip()
    # Keep common acronyms uppercased
    # But we avoid aggressive title-case for GPU names; still looks nicer than raw slugs.
    return re.sub(r"\s+", " ", t)

def shorten(s: str, max_len: int = 36) -> str:
    s = clean_text(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"

def get_first_literal(g: Graph, s: URIRef, p: URIRef) -> Optional[str]:
    for o in g.objects(s, p):
        if isinstance(o, Literal):
            return clean_text(str(o))
    return None

def get_rdfs_label(g: Graph, node: URIRef) -> Optional[str]:
    for o in g.objects(node, RDFS.label):
        if isinstance(o, Literal):
            return clean_text(str(o))
    return None

def get_entity_type(g: Graph, node: URIRef, base_pred_local: str = "entityType") -> Optional[str]:
    # In your TTL generator: ex:entityType "gpu" etc
    for p in g.predicates(node, None):
        if isinstance(p, URIRef) and local_name(str(p)) == base_pred_local:
            val = get_first_literal(g, node, p)
            if val:
                return val
    # Fallback to rdf:type local
    for o in g.objects(node, RDF.type):
        if isinstance(o, URIRef):
            return local_name(str(o))
    return None

def is_relation_node(uri: str) -> bool:
    return local_name(uri).startswith("rel_")

def is_metadata_predicate(p: URIRef, extra_ignored: Set[str]) -> bool:
    if p == RDF.type or p == RDFS.label:
        return True
    pl = local_name(str(p))
    default_ignored = {
        "sourceUrl", "sourceText", "entityType", "category",
        "subject", "object", "predicate"
    }
    return (pl in default_ignored) or (pl in extra_ignored)

def guess_node_group(entity_type: str) -> str:
    t = (entity_type or "").lower()
    if "gpu" in t:
        return "gpu"
    if "manufacturer" in t or "mfg" in t:
        return "manufacturer"
    if "architecture" in t or "arch" in t:
        return "architecture"
    if "memory" in t:
        return "memory_type"
    if "relation" in t:
        return "relation"
    return "other"


# -----------------------------
# Graph building (semantic edges only)
# -----------------------------

def build_semantic_edges(
    ttl_path: str,
    ignore_predicates: Set[str],
    include_relation_nodes: bool,
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """
    Returns:
      nodes: set of node IRIs (strings)
      edges: set of (src_iri, dst_iri, predicate_local)
    """
    rg = Graph()
    rg.parse(ttl_path, format="turtle")

    nodes: Set[str] = set()
    edges: Set[Tuple[str, str, str]] = set()

    for s, p, o in rg:
        if not (isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef)):
            continue

        s_id, o_id = str(s), str(o)
        if not include_relation_nodes and (is_relation_node(s_id) or is_relation_node(o_id)):
            continue

        if is_metadata_predicate(p, ignore_predicates):
            continue

        pred = local_name(str(p))
        nodes.add(s_id)
        nodes.add(o_id)
        edges.add((s_id, o_id, pred))

    return nodes, edges


# -----------------------------
# Visualization (HTML only)
# -----------------------------

def build_pyvis_network(
    ttl_path: str,
    nodes: Set[str],
    edges: Set[Tuple[str, str, str]],
    out_html: str,
    layout: str,
    label_max: int,
) -> None:
    rg = Graph()
    rg.parse(ttl_path, format="turtle")

    # Network init
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#0b1020",
        font_color="#e6e6e6",
        notebook=False
    )

    # Better physics + styling
    # - repulsion reduces overlap
    # - stabilization prevents jumpy final graph
    net.set_options(
        """
        var options = {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "multiselect": true,
            "tooltipDelay": 80
          },
          "nodes": {
            "shape": "dot",
            "size": 18,
            "borderWidth": 2,
            "shadow": true,
            "font": {"size": 16, "face": "arial"}
          },
          "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
            "smooth": {"type": "dynamic"},
            "shadow": false,
            "font": {"size": 14, "align": "top"},
            "color": {"inherit": false, "opacity": 0.6}
          },
          "physics": {
            "enabled": true,
            "stabilization": {"enabled": true, "iterations": 350},
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -80,
              "centralGravity": 0.01,
              "springLength": 140,
              "springConstant": 0.08,
              "avoidOverlap": 1
            },
            "maxVelocity": 40,
            "minVelocity": 0.1
          }
        }
        """
    )

    # Color palette (fixed, readable, not random)
    colors = {
        "manufacturer": {"background": "#ffb703", "border": "#c77d00"},
        "gpu": {"background": "#219ebc", "border": "#0b6b85"},
        "architecture": {"background": "#8ecae6", "border": "#4a90b5"},
        "memory_type": {"background": "#90be6d", "border": "#4f7c3a"},
        "relation": {"background": "#adb5bd", "border": "#6c757d"},
        "other": {"background": "#b5179e", "border": "#7209b7"},
    }

    # Build nodes with labels + tooltips
    for n in nodes:
        node_uri = URIRef(n)
        label = get_rdfs_label(rg, node_uri)
        if not label:
            label = pretty_local(local_name(n))

        entity_type = get_entity_type(rg, node_uri) or "Thing"
        group = guess_node_group(entity_type)

        c = colors.get(group, colors["other"])
        title_lines = [
            f"<b>Label:</b> {clean_text(label)}",
            f"<b>Type:</b> {clean_text(entity_type)}",
            f"<b>IRI:</b> {clean_text(n)}",
        ]

        # Optional metadata
        # If present, show category/sourceUrl
        # (We don't depend on exact prefix; we search by local name.)
        for p in rg.predicates(node_uri, None):
            if isinstance(p, URIRef):
                pl = local_name(str(p))
                if pl in {"category", "sourceUrl"}:
                    val = get_first_literal(rg, node_uri, p)
                    if val:
                        title_lines.append(f"<b>{pl}:</b> {val}")

        net.add_node(
            n,
            label=shorten(label, label_max),
            title="<br/>".join(title_lines),
            color=c,
        )

    # Add edges
    for s, o, pred in edges:
        net.add_edge(
            s,
            o,
            label=pred,
            title=pred,
        )

    # Layout: hierarchical (nice for your KG) or physics
    if layout == "hierarchical":
        # Hierarchical is great for Manufacturer → GPU → (Architecture/Memory)
        net.set_options(
            """
            var options = {
              "layout": {
                "hierarchical": {
                  "enabled": true,
                  "direction": "LR",
                  "sortMethod": "hubsize",
                  "levelSeparation": 220,
                  "nodeSpacing": 160
                }
              },
              "physics": {"enabled": false}
            }
            """
        )

    # Enable filter/search UI in PyVis (shows buttons)
    net.show_buttons(filter_=["physics", "layout", "interaction", "nodes", "edges"])

    net.write_html(out_html)
    print(f"[OK] Saved improved HTML: {out_html}")


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttl", required=True, help="Path to TTL file (e.g., gpu_kg.ttl)")
    ap.add_argument("--html", default="", help="Output HTML path (default: <ttl>.pretty.html)")
    ap.add_argument("--layout", choices=["physics", "hierarchical"], default="physics",
                    help="physics = forceAtlas2Based; hierarchical = LR tree-like layout")
    ap.add_argument("--label-max", type=int, default=40, help="Max node label length")
    ap.add_argument("--include-relation-nodes", action="store_true",
                    help="Include ex:rel_* relation nodes (usually clutter)")
    ap.add_argument("--ignore-predicates", default="",
                    help="Comma-separated predicate local names to ignore (in addition to defaults)")
    args = ap.parse_args()

    ttl_path = str(Path(args.ttl))
    out_html = args.html.strip() or str(Path(ttl_path).with_suffix(".pretty.html"))
    ignore = {x.strip() for x in args.ignore_predicates.split(",") if x.strip()}

    nodes, edges = build_semantic_edges(
        ttl_path=ttl_path,
        ignore_predicates=ignore,
        include_relation_nodes=args.include_relation_nodes,
    )

    if not nodes or not edges:
        raise SystemExit(
            "Graph is empty after filtering. Try --include-relation-nodes or adjust --ignore-predicates."
        )

    build_pyvis_network(
        ttl_path=ttl_path,
        nodes=nodes,
        edges=edges,
        out_html=out_html,
        layout=args.layout,
        label_max=args.label_max,
    )


if __name__ == "__main__":
    main()