"""
predict.py - Predict VTuber Recommendations Based on Tag Preferences

This module analyzes VTuber streaming data to recommend creators based on tag preferences.
It combines content-based scoring (tag affinity) with social influence (collaboration network)
to rank VTubers for given query tags.

The scoring pipeline:
1. Content Score: Based on historical gameplay patterns matching query tags
2. Social Score: Based on collaboration connections to high-scoring VTubers
3. Final Score: Weighted combination with propagation boost from centrality metrics

Usage:
    python predict.py --graphs graphs/graphs/ --tags game_tags.json --ontology ontology.txt --query "fps,multiplayer,competitive" --alpha 0.8





    
    # Recommend VTubers for FPS games
    python predict.py --graphs output/graphs/ --tags game_tags.json --ontology ontology.txt --query fps --top 10

    # Recommend VTubers for competitive multiplayer with org filter
    python predict.py --graphs output/graphs/ --tags game_tags.json --ontology ontology.txt \
        --query "competitive,multiplayer" --org-allowlist "Hololive,NIJISANJI" --top 15

    # Use only content-based scoring (no social influence)
    python predict.py --graphs output/graphs/ --tags game_tags.json --ontology ontology.txt \
        --query horror --algo tag_only

    # Adjust weighting parameters
    python predict.py --graphs output/graphs/ --tags game_tags.json --ontology ontology.txt \
        --query fps --alpha 0.8 --collab-mix 0.7 --beta 0.4

Output:
    Ranked table of VTubers with scores for content affinity, social influence,
    centrality metrics, and tag matching percentages.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx


def load_json(fp: Path) -> Any:
    """
    Load and parse a JSON file.

    Args:
        fp: Path to the JSON file.

    Returns:
        The parsed JSON content.
    """
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ontology(fp: Path) -> list[str]:
    """
    Load tag ontology from a text file.

    Expects one tag per line, ignoring empty lines and whitespace.

    Args:
        fp: Path to the ontology text file.

    Returns:
        List of tag strings.
    """
    tags = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tags.append(line)
    return tags


def parse_iso8601_z(s: str) -> datetime:
    """
    Parse an ISO 8601 datetime string, handling the 'Z' suffix for UTC.

    Args:
        s: Datetime string, optionally ending with 'Z'.

    Returns:
        A timezone-aware datetime object.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


@dataclass(frozen=True)
class PredictConfig:
    """
    Configuration for the prediction/recommendation algorithm.

    Attributes:
        top_k: Number of top VTubers to return in results.
        algo: Scoring algorithm - "linear" (default), "tag_only", or "social_only".
        alpha: Weight for content vs social influence in linear mode (0.7 = 70% content).
        collab_mix: Mix between count_weight and duration_weight for collab edges (0.5).
        inactive_cutoff_days: Filter out VTubers inactive longer than this (None = no filter).
        beta: Bounded propagation boost multiplier (max = 1 + beta).
        lambda_eig: Mix between eigenvector and weighted-degree in propagation (0.7).
        purity_bonus: Boost for VTubers specializing in query tags (0.25).
        tag_coverage_bonus: Boost for VTubers covering multiple query tags (0.5).
        synergy_bonus: Boost for VTubers playing games matching ALL query tags (1.0).
        community_method: Community detection algorithm - "leiden" or "louvain".
        eigen_max_iter: Maximum iterations for eigenvector centrality calculation.
        eigen_tol: Tolerance for eigenvector centrality convergence.
        org_allowlist: If set, only include VTubers from these organizations.
        include_supporting_analyses: Whether to print community/centrality analyses.
    """
    top_k: int = 5
    algo: str = "linear"
    alpha: float = 0.7
    collab_mix: float = 0.5
    inactive_cutoff_days: float | None = 90.0
    beta: float = 0.55
    lambda_eig: float = 0.70
    purity_bonus: float = 0.25
    tag_coverage_bonus: float = 0.50
    synergy_bonus: float = 1.00
    community_method: str = "leiden"
    eigen_max_iter: int = 500
    eigen_tol: float = 1e-6
    org_allowlist: tuple[str, ...] | None = None
    include_supporting_analyses: bool = True


def load_last_seen(last_seen_path: Path) -> dict[str, datetime]:
    """
    Load last_seen timestamps from JSON file.

    Args:
        last_seen_path: Path to last_seen.json with vtuber_id -> isoformat timestamps.

    Returns:
        Dictionary mapping VTuber IDs to aware datetime objects.
    """
    obj = load_json(last_seen_path)
    out = {}
    for k, v in obj.items():
        out[k] = datetime.fromisoformat(v).astimezone(timezone.utc)
    return out


def _minmax_norm(values: dict[str, float]) -> dict[str, float]:
    """
    Normalize values to [0, 1] range using min-max scaling.

    Args:
        values: Dictionary mapping keys to numeric values.

    Returns:
        Dictionary with values scaled to [0, 1]. Returns empty dict if input is empty.
        All values map to 0.0 if max equals min.
    """
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    if vmax <= vmin:
        return {k: 0.0 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}


def _minmax_0_1(scores: dict[str, float]) -> dict[str, float]:
    """
    Normalize scores to [0, 1] range, alias for _minmax_norm.
    """
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _safe_div(a: float, b: float) -> float:
    """Safe division returning 0.0 when divisor is zero."""
    return a / b if b != 0.0 else 0.0


def load_graphs(vtuber_game_pickle: Path, collab_pickle: Path) -> tuple[nx.Graph, nx.Graph]:
    """
    Load the two graphs from pickle files.

    Args:
        vtuber_game_pickle: Path to the bipartite VTuber-game graph.
        collab_pickle: Path to the VTuber collaboration graph.

    Returns:
        Tuple of (B, C) where B is bipartite graph and C is collaboration graph.
    """
    with open(vtuber_game_pickle, "rb") as f:
        B = pickle.load(f)
    with open(collab_pickle, "rb") as f:
        C = pickle.load(f)
    return B, C


def load_game_tags(game_tags_path: Path) -> dict:
    """
    Load game tags mapping from JSON file.

    Args:
        game_tags_path: Path to game_tags.json.

    Returns:
        Dictionary mapping topic_id to entry with 'tags' and 'is_game' fields.
    """
    return load_json(game_tags_path)


def tag_score_from_focus_likelihood_multiplicative(
    any_abs: dict[str, float],
    any_frac: dict[str, float],
    all_abs: dict[str, float],
    coverage: dict[str, float],
    *,
    purity_bonus: float = 0.25,
    tag_coverage_bonus: float = 0.50,
    synergy_bonus: float = 1.00,
) -> dict[str, float]:
    """
    Compute content scores using multiplicative tag matching semantics.

    For query tags "fps, competitive":
      - Match VTubers who play fps OR competitive (volume base)
      - Prefer VTubers covering BOTH tags (coverage)
      - Strongly prefer VTubers playing games with BOTH tags simultaneously (synergy)

    Formula:
      score = base_any * (1 + purity_bonus * any_frac)
                     * (1 + tag_coverage_bonus * coverage)
                     * (1 + synergy_bonus * base_all)

    Args:
        any_abs: Sum of weights for games matching ANY query tag.
        any_frac: Fraction of total activity in ANY-matching games.
        all_abs: Sum of weights for games matching ALL query tags.
        coverage: Tag coverage score (0-1) balancing breadth and depth.
        purity_bonus: Boost for specialization in query tags.
        tag_coverage_bonus: Boost for covering multiple query tags.
        synergy_bonus: Boost for games matching all tags simultaneously.

    Returns:
        Dictionary mapping VTuber IDs to content scores (0-1 range).
    """
    vol_any = {vt: math.log1p(max(0.0, v)) for vt, v in any_abs.items()}
    vol_all = {vt: math.log1p(max(0.0, v)) for vt, v in all_abs.items()}

    base_any = _minmax_norm(vol_any)
    base_all = _minmax_norm(vol_all) if vol_all else {}

    out: dict[str, float] = {}
    keys = set(base_any) | set(any_frac) | set(coverage) | set(base_all)

    for vt in keys:
        b_any = float(base_any.get(vt, 0.0))
        p = 1.0 + float(purity_bonus) * float(any_frac.get(vt, 0.0))
        c = 1.0 + float(tag_coverage_bonus) * float(coverage.get(vt, 0.0))
        s = 1.0 + float(synergy_bonus) * float(base_all.get(vt, 0.0))
        out[vt] = b_any * p * c * s

    return _minmax_norm(out)


def compute_focus_activity_multi(
    B: nx.Graph,
    game_tags: dict,
    query_tags: list[str],
    min_tag_share: float = 0.02,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """
    Compute activity metrics for query tags across VTuber-game bipartite graph.

    Args:
        B: The bipartite VTuber-game graph.
        game_tags: Mapping of topic_id to tag entries.
        query_tags: List of tags to search for.
        min_tag_share: Minimum fraction of activity for a tag to count (default 2%).

    Returns:
        Tuple of:
        - sumWeights_tag_ANY: Weight sum for games matching ANY query tag
        - sumWeights_tag_ALL: Weight sum for games matching ALL query tags
        - sumWeights_edge_EveryEdge: Total weight across all edges per VTuber
        - tag_coverage: Fraction of query tags with sufficient presence
    """
    q = [t.strip() for t in query_tags if t and t.strip()]
    qset = set(q)

    sumWeights_tag_ANY = {}
    sumWeights_tag_ALL = {}
    sumWeights_edge_EveryEdge = {}
    tag_coverage = {}

    per_tag_abs = {t: {} for t in q}

    for u, v, ed in B.edges(data=True):
        if u.startswith("vt:") and v.startswith("g:"):
            vt, g = u, v
        elif v.startswith("vt:") and u.startswith("g:"):
            vt, g = v, u
        else:
            continue

        w = float(ed.get("weight") or 0.0)
        if w <= 0:
            continue

        sumWeights_edge_EveryEdge[vt] = sumWeights_edge_EveryEdge.get(vt, 0.0) + w

        topic_id = g[2:]
        gt = game_tags.get(topic_id)
        if not gt or gt.get("is_game") is False:
            continue

        tags = set(gt.get("tags") or [])

        if tags & qset:
            sumWeights_tag_ANY[vt] = sumWeights_tag_ANY.get(vt, 0.0) + w

        if qset and qset.issubset(tags):
            sumWeights_tag_ALL[vt] = sumWeights_tag_ALL.get(vt, 0.0) + w

        for t in q:
            if t in tags:
                d = per_tag_abs[t]
                d[vt] = d.get(vt, 0.0) + w

    for vt, tot in sumWeights_edge_EveryEdge.items():
        if not q or tot <= 0:
            tag_coverage[vt] = 0.0
            continue

        tag_or = sumWeights_tag_ANY.get(vt, 0.0) / tot

        covered = 0
        for t in q:
            share_t = per_tag_abs[t].get(vt, 0.0) / tot
            if share_t >= min_tag_share:
                covered += 1

        thresholded_presence_fraction = covered / float(len(q))
        tag_coverage[vt] = tag_or * thresholded_presence_fraction

    return sumWeights_tag_ANY, sumWeights_tag_ALL, sumWeights_edge_EveryEdge, tag_coverage


def tag_score_from_focus_likelihood(
    focus_abs: dict[str, float],
    focus_frac: dict[str, float],
    purity_bonus: float = 0.25,
) -> dict[str, float]:
    """
    Simple content scoring (legacy, kept for reference).

    Uses log-compressed volume with a small purity boost.
    """
    vol = {vt: math.log1p(max(0.0, fa)) for vt, fa in focus_abs.items()}
    vol_norm = _minmax_norm(vol)

    out = {}
    keys = set(vol_norm) | set(focus_frac)
    for vt in keys:
        base = float(vol_norm.get(vt, 0.0))
        frac = float(focus_frac.get(vt, 0.0))
        out[vt] = base * (1.0 + purity_bonus * frac)
    return out


def build_vtuber_tag_profiles(
    B: nx.Graph,
    game_tags: dict,
    ontology: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """
    Build per-VTuber tag preference profiles from the bipartite graph.

    Args:
        B: The bipartite VTuber-game graph.
        game_tags: Mapping of topic_id to tag entries.
        ontology: List of valid tags (unknown tags are ignored).

    Returns:
        Tuple of:
        - vt_tag_weights: VTuber -> tag -> weight mapping
        - vt_total: Total tag weight per VTuber
    """
    ontology_set = set(ontology)
    vt_tag_weights: dict[str, dict[str, float]] = {}
    vt_total: dict[str, float] = {}

    for vt, g, ed in B.edges(data=True):
        if not (isinstance(vt, str) and vt.startswith("vt:")):
            vt, g = g, vt
        if not (isinstance(vt, str) and vt.startswith("vt:")):
            continue
        if not (isinstance(g, str) and g.startswith("g:")):
            continue

        topic_id = g[2:]
        gt = game_tags.get(topic_id)
        if not gt:
            continue
        if gt.get("is_game") is False:
            continue

        tags = gt.get("tags") or []
        if not isinstance(tags, list) or not tags:
            continue

        w = float(ed.get("weight") or 0.0)
        if w <= 0:
            continue

        for t in tags:
            t = str(t)
            if t not in ontology_set:
                continue
            vt_tag_weights.setdefault(vt, {})
            vt_tag_weights[vt][t] = vt_tag_weights[vt].get(t, 0.0) + w
            vt_total[vt] = vt_total.get(vt, 0.0) + w

    return vt_tag_weights, vt_total


def tag_influence_scores(
    vt_tag_weights: dict[str, dict[str, float]],
    vt_total: dict[str, float],
    query_tags: list[str],
) -> dict[str, float]:
    """
    Simple tag influence: fraction of tag mass in query tags.

    Args:
        vt_tag_weights: VTuber -> tag -> weight mapping.
        vt_total: Total weight per VTuber.
        query_tags: Tags to score against.

    Returns:
        Dictionary mapping VTuber IDs to influence scores (0-1).
    """
    q = set(query_tags)
    scores: dict[str, float] = {}
    for vt, tw in vt_tag_weights.items():
        num = sum(w for t, w in tw.items() if t in q)
        den = vt_total.get(vt, 0.0)
        scores[vt] = _safe_div(num, den)
    return scores


def _ensure_mixed_collab_weight(C: nx.Graph, collab_mix: float, *, out_attr: str = "w_mix") -> None:
    """
    Add a combined weight attribute to each collaboration edge.

    Mixes count_weight and duration_weight according to collab_mix parameter.

    Args:
        C: Collaboration graph (modified in-place).
        collab_mix: Weight for count vs duration (0 = pure duration, 1 = pure count).
        out_attr: Name of the output weight attribute.
    """
    for u, v, ed in C.edges(data=True):
        cw = float(ed.get("count_weight") or 0.0)
        dw = float(ed.get("duration_weight") or 0.0)
        ed[out_attr] = collab_mix * cw + (1.0 - collab_mix) * dw


def weighted_degree_scores(C: nx.Graph, *, weight_attr: str = "w_mix") -> dict[str, float]:
    """
    Compute weighted degree (sum of incident edge weights) for each node.

    Args:
        C: Collaboration graph.
        weight_attr: Edge attribute to use for weights.

    Returns:
        Dictionary mapping node IDs to weighted degree values.
    """
    return {n: float(w) for n, w in C.degree(weight=weight_attr)}


def eigenvector_centrality_scores(
    C: nx.Graph,
    *,
    weight_attr: str = "w_mix",
    max_iter: int = 500,
    tol: float = 1e-6,
) -> dict[str, float]:
    """
    Compute eigenvector centrality for undirected weighted graphs.

    Args:
        C: Collaboration graph.
        weight_attr: Edge attribute to use for weights.
        max_iter: Maximum iterations for power method.
        tol: Convergence tolerance.

    Returns:
        Dictionary mapping node IDs to centrality scores.
    """
    if C.number_of_nodes() == 0:
        return {}
    if C.number_of_edges() == 0:
        return {n: 0.0 for n in C.nodes()}
    try:
        return nx.eigenvector_centrality(C, weight=weight_attr, max_iter=max_iter, tol=tol)
    except Exception:
        try:
            return nx.eigenvector_centrality_numpy(C, weight=weight_attr)
        except Exception:
            return {n: 0.0 for n in C.nodes()}


def community_assignments(
    C: nx.Graph,
    *,
    method: str = "leiden",
    weight_attr: str = "w_mix",
    seed: int = 42,
) -> dict[str, int]:
    """
    Detect communities in the collaboration graph.

    Args:
        C: Collaboration graph.
        method: "leiden" (recommended, uses igraph) or "louvain" (NetworkX).
        weight_attr: Edge attribute to use for weights.
        seed: Random seed for community detection.

    Returns:
        Dictionary mapping node IDs to community IDs.
    """
    if C.number_of_nodes() == 0:
        return {}
    if C.number_of_edges() == 0:
        return {n: 0 for n in C.nodes()}

    method = (method or "leiden").lower().strip()
    if method == "leiden":
        try:
            import igraph as ig
            import leidenalg

            nodes = list(C.nodes())
            idx = {n: i for i, n in enumerate(nodes)}
            edges = [(idx[u], idx[v]) for u, v in C.edges()]
            weights = [float(C[u][v].get(weight_attr) or 0.0) for u, v in C.edges()]

            g = ig.Graph(n=len(nodes), edges=edges, directed=False)
            g.es["weight"] = weights

            part = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights=g.es["weight"],
                seed=seed,
            )
            memb = list(part.membership)
            return {nodes[i]: int(memb[i]) for i in range(len(nodes))}
        except Exception:
            method = "louvain"

    if method == "louvain":
        try:
            from networkx.algorithms.community import louvain_communities

            comms = louvain_communities(C, weight=weight_attr, seed=seed)
            out: dict[str, int] = {}
            for cid, comm in enumerate(comms):
                for n in comm:
                    out[n] = cid
            return out
        except Exception:
            return {n: 0 for n in C.nodes()}

    return {n: 0 for n in C.nodes()}


def print_supporting_analyses(
    *,
    rows: list[dict],
    comm: dict[str, int],
    wdeg_0_1: dict[str, float],
    eig_0_1: dict[str, float],
    name_lookup: dict[str, str] | None = None,
    top_comm_k: int = 5,
    top_in_comm_k: int = 5,
) -> None:
    """
    Print supporting graph analyses: community detection and centrality metrics.

    Args:
        rows: Top-K ranked VTuber rows.
        comm: Community assignments for all VTubers.
        wdeg_0_1: Normalized weighted degree scores.
        eig_0_1: Normalized eigenvector centrality scores.
        name_lookup: ID -> English name mapping for display.
        top_comm_k: Number of largest communities to list.
        top_in_comm_k: Number of leaders to show per community.
    """
    print("\nSupporting analyses")
    print("-------------------")

    comm_counts = Counter(comm.values())
    print(f"Communities detected: {len(comm_counts)}")

    top_comms = comm_counts.most_common(top_comm_k)
    comm_summary = ", ".join(f"{cid} (n={cnt})" for cid, cnt in top_comms)
    print(f"Largest communities: {comm_summary}")

    id_to_name: dict[str, str] = {}

    for r in rows:
        rid = r.get("node_id") or r.get("vtuber_id") or r.get("id")
        rname = r.get("english_name") or r.get("vtuber") or r.get("name")
        if rid and rname:
            id_to_name[str(rid)] = str(rname)

    if name_lookup:
        for k, v in name_lookup.items():
            if k and v:
                id_to_name.setdefault(str(k), str(v))

    if top_comms:
        largest_cid = top_comms[0][0]
        members = [vt for vt, cid in comm.items() if cid == largest_cid]

        leaders = sorted(
            members,
            key=lambda vt: eig_0_1.get(vt, 0.0),
            reverse=True
        )[:top_in_comm_k]

        leader_names = []
        for vt in leaders:
            name = id_to_name.get(vt, vt)
            leader_names.append(f"{name} ({eig_0_1.get(vt, 0.0):.2f})")

        if leader_names:
            print(f"Most influential in community {largest_cid}: " + ", ".join(leader_names))


def social_influence_scores(
    C: nx.Graph,
    base_scores: dict[str, float],
    collab_mix: float,
) -> dict[str, float]:
    """
    Compute social influence as weighted neighbor average.

    Each VTuber's social score is the weighted average of their collaborators'
    base scores, where weights come from collaboration edge strength.

    Args:
        C: Collaboration graph.
        base_scores: Content scores to propagate.
        collab_mix: Mix between count and duration weights.

    Returns:
        Dictionary mapping VTuber IDs to social influence scores.
    """
    out: dict[str, float] = {}
    for vt in C.nodes():
        nbrs = list(C.neighbors(vt))
        if not nbrs:
            out[vt] = 0.0
            continue

        weights: list[tuple[str, float]] = []
        total_w = 0.0
        for nb in nbrs:
            ed = C.get_edge_data(vt, nb) or {}
            cw = float(ed.get("count_weight") or 0.0)
            dw = float(ed.get("duration_weight") or 0.0)
            w = collab_mix * cw + (1.0 - collab_mix) * dw
            if w <= 0:
                continue
            weights.append((nb, w))
            total_w += w

        if total_w <= 0:
            out[vt] = 0.0
            continue

        s = 0.0
        for nb, w in weights:
            s += (w / total_w) * float(base_scores.get(nb, 0.0))
        out[vt] = s

    return out


def combine_scores(
    tag_scores: dict[str, float],
    social_scores: dict[str, float],
    cfg: PredictConfig,
) -> dict[str, float]:
    """
    Combine content and social scores according to algorithm.

    Args:
        tag_scores: Content-based scores.
        social_scores: Social influence scores.
        cfg: Configuration with algo and alpha settings.

    Returns:
        Combined scores normalized to [0, 1].
    """
    if cfg.algo == "tag_only":
        raw = dict(tag_scores)
    elif cfg.algo == "social_only":
        raw = dict(social_scores)
    elif cfg.algo == "linear":
        raw = {}
        keys = set(tag_scores) | set(social_scores)
        for k in keys:
            raw[k] = cfg.alpha * float(tag_scores.get(k, 0.0)) + (1.0 - cfg.alpha) * float(social_scores.get(k, 0.0))
    else:
        raise ValueError(f"Unknown algo: {cfg.algo}")
    return _minmax_norm(raw)


def rank_vtubers(
    B: nx.Graph,
    scores: dict[str, float],
    content_scores_0_1: dict[str, float],
    social_scores_0_1: dict[str, float],
    top_k: int,
    *,
    focus_abs: dict[str, float] | None = None,
    focus_frac: dict[str, float] | None = None,
    tag_cov_0_1: dict[str, float] | None = None,
    tag_syn_0_1: dict[str, float] | None = None,
    allowed_orgs: set[str] | None = None,
    active_vtubers: set[str] | None = None,
    wdeg_0_1: dict[str, float] | None = None,
    eig_0_1: dict[str, float] | None = None,
    comm: dict[str, int] | None = None,
) -> list[dict]:
    """
    Build ranked results table for VTubers.

    Args:
        B: Bipartite graph (for node attributes).
        scores: Final combined scores.
        content_scores_0_1: Normalized content scores.
        social_scores_0_1: Normalized social scores.
        top_k: Maximum number of results to return.
        focus_abs: Raw tag-matching weights.
        focus_frac: Fraction of activity in matching games.
        tag_cov_0_1: Tag coverage scores.
        tag_syn_0_1: Tag synergy scores.
        allowed_orgs: Filter by organization.
        active_vtubers: Filter by recency.
        wdeg_0_1: Normalized weighted degree.
        eig_0_1: Normalized eigenvector centrality.
        comm: Community assignments.

    Returns:
        List of dictionaries with VTuber info and scores.
    """
    vtuber_nodes = [
        n for n, d in B.nodes(data=True)
        if d.get("kind") == "vtuber" and isinstance(n, str) and n.startswith("vt:")
    ]

    rows: list[dict] = []
    for vtuber in vtuber_nodes:
        if active_vtubers is not None and vtuber not in active_vtubers:
            continue

        d = B.nodes[vtuber]
        org = d.get("org")
        if allowed_orgs is not None and org not in allowed_orgs:
            continue

        rows.append({
            "vtuber": d.get("english_name") or d.get("name") or vtuber[3:],
            "channel_id": d.get("channel_id") or vtuber[3:],
            "score_0_1": float(scores.get(vtuber, 0.0)),
            "content_0_1": float(content_scores_0_1.get(vtuber, 0.0)),
            "social_0_1": float(social_scores_0_1.get(vtuber, 0.0)),
            "wdeg_0_1": float((wdeg_0_1 or {}).get(vtuber, 0.0)),
            "eig_0_1": float((eig_0_1 or {}).get(vtuber, 0.0)),
            "community": int((comm or {}).get(vtuber, 0)),
            "node_id": vtuber,
            "tag_volume": float((focus_abs or {}).get(vtuber, 0.0)),
            "tag_fraction": float((focus_frac or {}).get(vtuber, 0.0)),
            "tag_cov": float((tag_cov_0_1 or {}).get(vtuber, 0.0)),
            "tag_syn": float((tag_syn_0_1 or {}).get(vtuber, 0.0)),
            "org": org,
            "suborg": d.get("suborg"),
        })

    rows.sort(key=lambda r: r["score_0_1"], reverse=True)
    return rows[:top_k]


def print_pretty_header(query_tags: list[str], cfg: PredictConfig) -> None:
    """Print formatted query header."""
    tags = ", ".join(query_tags)
    print(f"Query tags: {tags}")
    print(f"Top-K: {cfg.top_k} | Algo: {cfg.algo} | alpha(content): {cfg.alpha:.2f} | collab_mix: {cfg.collab_mix:.2f}")


def print_table(rows: list[dict]) -> None:
    """
    Print formatted results table with grouped headers.

    Columns: Rank, VTuber, Score, Content, Social, Deg, Eig, Community, Tag_OR, Tag_AND
    """
    headers = [
        "Rank", "VTuber", "Score", "Content", "Social", "Deg", "Eig", "Community", "Tag_OR", "Tag_AND",
    ]

    cols = [
        [str(i + 1) for i in range(len(rows))],
        [r["vtuber"] for r in rows],
        [f'{r["score_0_1"]:.3f}' for r in rows],
        [f'{r["content_0_1"]:.3f}' for r in rows],
        [f'{r["social_0_1"]:.3f}' for r in rows],
        [f'{r["wdeg_0_1"]:.3f}' for r in rows],
        [f'{r["eig_0_1"]:.3f}' for r in rows],
        [str(r["community"]) for r in rows],
        [f'{r["tag_fraction"] * 100:.1f}%' for r in rows],
        [f'{r["tag_syn"] * 100:.1f}%' for r in rows],
    ]

    sep = "  "
    widths = [
        max(len(h), max((len(v) for v in col), default=0))
        for h, col in zip(headers, cols)
    ]

    def fmt_row(vals: list[str]) -> str:
        return sep.join(v.ljust(w) for v, w in zip(vals, widths))

    header_line = fmt_row(headers)

    starts = []
    pos = 0
    for w in widths:
        starts.append(pos)
        pos += w + len(sep)
    total_len = len(header_line)

    top_chars = list(" " * total_len)

    def overlay(label: str, c0: int, c1: int) -> None:
        left = starts[c0]
        right = starts[c1] + widths[c1]
        span = right - left
        lab = label[:span]
        off = left + max(0, (span - len(lab)) // 2)
        for i, ch in enumerate(lab):
            if 0 <= off + i < total_len:
                top_chars[off + i] = ch

    overlay("Centrality", 5, 6)
    overlay("Tag Matching", 8, 9)

    top_line = "".join(top_chars).rstrip()

    if top_line:
        print(top_line)
    print(header_line)
    print(sep.join("─" * w for w in widths))
    for i in range(len(rows)):
        print(fmt_row([col[i] for col in cols]))

    print()
    print(
        "Score / Content / Social / Deg / Eig are relative (normalized across VTubers). "
        "Tag_OR and Tag_AND are absolute percentages of this VTuber's activity."
    )


def run_query(
    *,
    vtuber_game_pickle: Path,
    collab_pickle: Path,
    ontology_path: Path,
    game_tags_path: Path,
    query_tags: list[str],
    cfg: PredictConfig,
) -> None:
    """
    Execute full recommendation pipeline and print results.

    Args:
        vtuber_game_pickle: Path to bipartite graph pickle.
        collab_pickle: Path to collaboration graph pickle.
        ontology_path: Path to ontology.txt.
        game_tags_path: Path to game_tags.json.
        query_tags: Tags to search for.
        cfg: Configuration object.
    """
    ontology = load_ontology(ontology_path)
    ontology_set = set(ontology)
    bad = [tag for tag in query_tags if tag not in ontology_set]
    if bad:
        raise ValueError(f"Unknown tags not in ontology: {bad}")

    B, C = load_graphs(vtuber_game_pickle, collab_pickle)
    game_tags = load_game_tags(game_tags_path)

    active_vtubers: set[str] | None = None
    if cfg.inactive_cutoff_days is not None:
        artifacts_dir = vtuber_game_pickle.parent.parent / "artifacts"

        meta = load_json(artifacts_dir / "build_metadata.json")
        now_utc = datetime.fromisoformat(meta["now_utc"]).astimezone(timezone.utc)

        last_seen = load_last_seen(artifacts_dir / "last_seen.json")
        cutoff_seconds = float(cfg.inactive_cutoff_days) * 24.0 * 3600.0

        active_vtubers = set()
        for vtuber, last_seen_at in last_seen.items():
            if (now_utc - last_seen_at).total_seconds() <= cutoff_seconds:
                active_vtubers.add(vtuber)

    allowed_orgs = set(cfg.org_allowlist) if cfg.org_allowlist else None

    def _keep(vtuber: str) -> bool:
        if active_vtubers is not None and vtuber not in active_vtubers:
            return False
        if allowed_orgs is not None:
            d = B.nodes.get(vtuber, {})
            if d.get("org") not in allowed_orgs:
                return False
        return True

    sumWeights_tag_ANY, sumWeights_tag_ALL, sumWeights_edge_EveryEdge, tag_coverage = \
        compute_focus_activity_multi(B, game_tags, query_tags, 0.02)

    tag_OR = {
        vtuber: (sumWeights_tag_ANY.get(vtuber, 0.0) / sumWeights_edge_EveryEdge[vtuber])
        if sumWeights_edge_EveryEdge.get(vtuber, 0.0) > 0 else 0.0
        for vtuber in sumWeights_edge_EveryEdge
    }
    tag_and_abs = {
        vtuber: (sumWeights_tag_ALL.get(vtuber, 0.0) / sumWeights_edge_EveryEdge.get(vtuber, 1.0))
        if sumWeights_edge_EveryEdge.get(vtuber, 0.0) > 0 else 0.0
        for vtuber in sumWeights_edge_EveryEdge
    }

    content_score = tag_score_from_focus_likelihood_multiplicative(
        sumWeights_tag_ANY,
        tag_OR,
        sumWeights_tag_ALL,
        tag_coverage,
        purity_bonus=cfg.purity_bonus,
        tag_coverage_bonus=cfg.tag_coverage_bonus,
        synergy_bonus=cfg.synergy_bonus,
    )
    content_score = {vtuber: v for vtuber, v in content_score.items() if _keep(vtuber)}

    social_score = social_influence_scores(C, content_score, cfg.collab_mix)
    social_score = {vtuber: v for vtuber, v in social_score.items() if _keep(vtuber)}

    if cfg.algo == "tag_only":
        base_raw = dict(content_score)
    elif cfg.algo == "social_only":
        base_raw = dict(social_score)
    elif cfg.algo == "linear":
        base_raw = {}
        keys = set(content_score) | set(social_score)
        for k in keys:
            base_raw[k] = cfg.alpha * float(content_score.get(k, 0.0)) + (1.0 - cfg.alpha) * float(social_score.get(k, 0.0))
    else:
        raise ValueError(f"Unknown algo: {cfg.algo}")
    base_raw = {vtuber: v for vtuber, v in base_raw.items() if _keep(vtuber)}
    nodes_kept = set(base_raw.keys())
    C_view = C.subgraph(nodes_kept).copy()
    _ensure_mixed_collab_weight(C_view, cfg.collab_mix, out_attr="w_mix")

    wdeg_raw = weighted_degree_scores(C_view, weight_attr="w_mix")
    eig_raw = eigenvector_centrality_scores(
        C_view,
        weight_attr="w_mix",
        max_iter=cfg.eigen_max_iter,
        tol=cfg.eigen_tol,
    )

    comm = community_assignments(
        C_view,
        method=cfg.community_method,
        weight_attr="w_mix",
    )

    content_0_1 = _minmax_norm(content_score)
    social_0_1 = _minmax_norm(social_score)
    wdeg_0_1 = _minmax_0_1(wdeg_raw)
    eig_0_1 = _minmax_0_1(eig_raw)

    beta = float(cfg.beta)
    lambda_eig = float(cfg.lambda_eig)
    final_raw = {}
    for vtuber, base in base_raw.items():
        net = beta * (lambda_eig * float(eig_0_1.get(vtuber, 0.0)) + (1.0 - lambda_eig) * float(wdeg_0_1.get(vtuber, 0.0)))
        final_raw[vtuber] = float(base) * (1.0 + net)

    final_0_1 = _minmax_norm(final_raw)

    tags = ", ".join(query_tags)
    print(f"Query tags: {tags}")
    print(f"Top-K: {cfg.top_k} | Algo: {cfg.algo} | alpha(content): {cfg.alpha:.2f} | collab_mix: {cfg.collab_mix:.2f}")
    if allowed_orgs is not None:
        print(f"Org filter: {', '.join(sorted(allowed_orgs))}")
    if active_vtubers is not None and cfg.inactive_cutoff_days is not None:
        print(f"Active cutoff: {float(cfg.inactive_cutoff_days):.0f} days")
    print()

    rows = rank_vtubers(
        B,
        scores=final_0_1,
        content_scores_0_1=content_0_1,
        social_scores_0_1=social_0_1,
        top_k=cfg.top_k,
        focus_abs=sumWeights_tag_ANY,
        focus_frac=tag_OR,
        tag_cov_0_1=_minmax_norm(tag_coverage),
        tag_syn_0_1=tag_and_abs,
        allowed_orgs=allowed_orgs,
        active_vtubers=active_vtubers,
        wdeg_0_1=wdeg_0_1,
        eig_0_1=eig_0_1,
        comm=comm,
    )

    print_table(rows)

    if cfg.include_supporting_analyses:
        print_supporting_analyses(
            rows=rows,
            comm=comm,
            wdeg_0_1=wdeg_0_1,
            eig_0_1=eig_0_1,
            name_lookup={vtuber: (B.nodes.get(vtuber, {}).get("english_name") or B.nodes.get(vtuber, {}).get("name") or vtuber) for vtuber in nodes_kept},
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the recommendation tool.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Predict VTuber recommendations based on tag preferences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--graphs",
        type=Path,
        required=True,
        help="Directory containing graph files (expects vtuber_game.gpickle and collab.gpickle)",
    )
    parser.add_argument(
        "--tags",
        type=Path,
        required=True,
        help="Path to game_tags.json",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        required=True,
        help="Path to ontology.txt (list of valid tags)",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query tags (comma-separated, e.g., 'fps,horror')",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top results to display",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="linear",
        choices=["linear", "tag_only", "social_only"],
        help="Scoring algorithm",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Content weight in linear mode (0-1, higher = more content-based)",
    )
    parser.add_argument(
        "--collab-mix",
        type=float,
        default=0.5,
        help="Mix count vs duration for collab edges (0=duration, 1=count)",
    )
    parser.add_argument(
        "--org-allowlist",
        type=str,
        default=None,
        help="Comma-separated organizations to include (e.g., 'Hololive,NIJISANJI')",
    )
    parser.add_argument(
        "--inactive-cutoff",
        type=float,
        default=90,
        help="Filter out VTubers inactive longer than this many days (0 to disable)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.55,
        help="Bounded propagation boost multiplier",
    )
    parser.add_argument(
        "--community",
        type=str,
        default="leiden",
        choices=["leiden", "louvain"],
        help="Community detection algorithm",
    )
    parser.add_argument(
        "--no-analyses",
        action="store_true",
        help="Skip supporting analyses (community, centrality)",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the prediction CLI.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    args = parse_args()

    vtuber_game_pickle = args.graphs / "vtuber_game.pickle"
    collab_pickle = args.graphs / "collab.pickle"

    if not vtuber_game_pickle.exists():
        print(f"Error: Graph file not found: {vtuber_game_pickle}", file=sys.stderr)
        return 1
    if not collab_pickle.exists():
        print(f"Error: Graph file not found: {collab_pickle}", file=sys.stderr)
        return 1

    query_tags = [t.strip() for t in args.query.split(",") if t.strip()]

    cfg = PredictConfig(
        top_k=args.top,
        algo=args.algo,
        alpha=args.alpha,
        collab_mix=args.collab_mix,
        inactive_cutoff_days=args.inactive_cutoff if args.inactive_cutoff > 0 else None,
        beta=args.beta,
        community_method=args.community,
        org_allowlist=tuple(args.org_allowlist.split(",")) if args.org_allowlist else None,
        include_supporting_analyses=not args.no_analyses,
    )

    try:
        run_query(
            vtuber_game_pickle=vtuber_game_pickle,
            collab_pickle=collab_pickle,
            ontology_path=args.ontology,
            game_tags_path=args.tags,
            query_tags=query_tags,
            cfg=cfg,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
