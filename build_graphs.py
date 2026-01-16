"""
TODO: decide whether to put all edge weighting in here, or in future analysis code. Currently we're doing edge weighting and merging recencyweight+softcapweight -> weight for B graph here, but not merging weights for C graph here; pick one and stick to it!
build_graphs.py - Build VTuber Collaboration and Game Interaction Graphs

This code builds two NetworkX graphs from a directory of JSON files containing stream data.
1) Bipartite graph B: VTuber ↔ Game     (vtubers and the games they played)
    VTuber nodes are labeled vt:<channel_id>
    Game nodes are labeled g:<topic_id>
    Edge weight aggregates per stream as:
        weight += recencyweight * softcapped_duration(duration_seconds)
            recencyweight (normalized 0..1) = 0.5 ^ (days_since / half_life_days)
            softcapped_duration = stream duration in seconds, up to parameter "soft-cap-seconds"; sqrt compression of any extra time over the cap
2) Collaboration graph C: VTuber ↔ VTuber   (vtubers who streamed together)
        For each collab partner, it aggregates:
            count_weight += recencyweight(collaboration occurrences)
            duration_weight += recencyweight * softcapped_duration(collab_duration_seconds)
                recencyweight (normalized 0..1) = 0.5 ^ (days_since / half_life_days)
                softcapped_duration = stream duration in seconds, up to parameter "soft-cap-seconds"; sqrt compression of any extra time over the cap

It also writes metadata and a last_seen.json mapping vtuber node → most recent stream start time.

The graphs are saved to out_dir/graphs.
Metadata saved to out_dir/artifacts.

Note: time "now" is computed from the most recent stream start time, for reproducibility for a given dataset at any real-world time.
---

Usage:
    python build_graphs.py --data-dir streams/ --denylist denylist.txt --out-dir graphs/ --half-life-days 180 --soft-cap-seconds 18000

Arguments:
    --data-dir: Directory containing JSON files with stream data (required)
    --denylist: Path to JSON file with topic IDs to exclude (optional)
    --out-dir: Directory for output files (default: output)
    --half-life-days: Recency weighting half-life in days (default: 180)
    --soft-cap-seconds: Maximum stream duration for weight calculation before starting to scale back its effect (default: 18000 = 5 hours)
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterator
import networkx as nx


def iter_json_files(data_dir: Path) -> Iterator[Path]:
    """
    Recursively find all JSON files in a directory.

    Args:
        data_dir: Directory to search. If a single JSON file is passed, yields it directly.

    Yields:
        Path objects for each JSON file found, sorted alphabetically.
    """
    if data_dir.is_file() and data_dir.suffix == ".json":
        yield data_dir
        return
    for f in sorted(data_dir.rglob("*.json")):
        yield f


def load_json(fp: Path) -> Any:
    """
    Load and parse a JSON file.

    Args:
        fp: Path to the JSON file.

    Returns:
        The parsed JSON content (can be dict, list, or any JSON-serializable type).
    """
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(fp: Path, obj: Any) -> None:
    """
    Write an object to a JSON file with pretty formatting.

    Args:
        fp: Path to the output file.
        obj: The object to serialize (must be JSON-serializable).
    """
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_denylist(fp: Path) -> set[str]:
    """
    Load a denylist of topic IDs to exclude from graph building.

    Supports both JSON and plain text formats:
    - JSON: List of IDs or dict where truthy values indicate exclusion
    - TXT: One ID per line (similar to ontology.txt format)

    Args:
        fp: Path to the denylist file.

    Returns:
        Set of topic ID strings to filter out.
    """
    if not fp.exists():
        return set()

    # Detect format by file extension
    if fp.suffix == ".txt":
        # Plain text format: one ID per line
        ids = set()
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(line)
        return ids

    # JSON format: list or dict
    data = load_json(fp)
    if isinstance(data, list):
        return {str(x) for x in data if x}
    if isinstance(data, dict):
        return {str(k) for k, v in data.items() if v}
    return set()


def parse_iso8601_z(s: str) -> datetime:
    """
    Parse an ISO 8601 datetime string, handling the 'Z' suffix for UTC.

    Args:
        s: Datetime string, optionally ending with 'Z' to indicate UTC.

    Returns:
        A timezone-aware datetime object.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


@dataclass
class TimeBounds:
    """
    Container for the temporal bounds of the dataset.

    Attributes:
        min_dt: The earliest stream start time in the dataset.
        max_dt: The latest stream start time in the dataset.
    """
    min_dt: datetime
    max_dt: datetime


def find_dataset_time_bounds(streams: list[dict]) -> TimeBounds:
    """
    Find the minimum and maximum timestamps across all stream records.

    Args:
        streams: List of stream dictionaries, each containing a 'start_actual' field.

    Returns:
        TimeBounds object with min_dt and max_dt. If no valid timestamps found,
        returns current UTC time for both bounds.
    """
    min_dt: datetime | None = None
    max_dt: datetime | None = None
    for s in streams:
        start = s.get("start_actual")
        if not start:
            continue
        try:
            dt = parse_iso8601_z(str(start))
        except ValueError:
            continue
        if min_dt is None or dt < min_dt:
            min_dt = dt
        if max_dt is None or dt > max_dt:
            max_dt = dt
    if min_dt is None or max_dt is None:
        now = datetime.now(timezone.utc)
        return TimeBounds(min_dt=now, max_dt=now)
    return TimeBounds(min_dt=min_dt, max_dt=max_dt)


@dataclass(frozen=True)
class WeightConfig:
    """
    Configuration for graph edge weight calculations.

    Attributes:
        half_life_days: Number of days for recency weight to decay to 50%.
        soft_cap_seconds: Duration cap for weight calculations; durations above
                         this are compressed using a square root function.
    """
    half_life_days: float = 180.0
    soft_cap_seconds: float = 5.0 * 3600.0


def softcap_duration(seconds: float, soft_cap_seconds: float) -> float:
    """
    Apply soft capping to a duration value for weight calculations.

    Durations at or below the cap are returned unchanged. Above the cap,
    the excess is compressed using square root to prevent outliers from
    dominating weights while still acknowledging higher engagement.

    Args:
        seconds: The raw duration in seconds.
        soft_cap_seconds: The cap threshold; values above this are compressed.

    Returns:
        The capped/compressed duration value.
    """
    if soft_cap_seconds <= 0:
        return seconds
    if seconds <= soft_cap_seconds:
        return seconds
    return soft_cap_seconds + math.sqrt(seconds - soft_cap_seconds)


def recency_weight(now_utc: datetime, start_utc: datetime, half_life_days: float) -> float:
    """
    Calculate a recency-based weight using exponential decay.

    Weights decay exponentially based on the time difference, with a configurable
    half-life. The formula is: weight = 0.5 ^ (days_since / half_life_days)

    Args:
        now_utc: The reference time (typically dataset max timestamp) in UTC.
        start_utc: The event timestamp in UTC.
        half_life_days: Days for weight to decay to 50% of original value. If set to 0, disable half-life decay (always return 1)

    Returns:
        Weight between 0.0 and 1.0, with 1.0 for events at or after now_utc.
    """
    delta = now_utc - start_utc
    days = delta.total_seconds() / 86400.0
    if days < 0:
        return 1.0
    if half_life_days <= 0:
        return 1.0
    return math.pow(0.5, days / half_life_days)


def stream_edge_weight(
    *,
    now_utc: datetime,
    start_utc: datetime,
    duration_seconds: float,
    cfg: WeightConfig,
) -> float:
    """
    Calculate the weight for a VTuber-game edge based on a stream.

    Combines recency weighting with soft-capped duration to produce
    an engagement score that emphasizes recent, substantial activity.

    Args:
        now_utc: Reference time for recency calculation.
        start_utc: Stream start time in UTC.
        duration_seconds: Raw stream duration.
        cfg: Weight configuration with half_life_days and soft_cap_seconds.

    Returns:
        Combined weight as product of recency and effective duration.
    """
    r = recency_weight(now_utc, start_utc, cfg.half_life_days)
    cdur_eff = softcap_duration(duration_seconds, cfg.soft_cap_seconds)
    return r * cdur_eff


@dataclass
class BuildOutputs:
    """
    Paths to files produced by the graph building process.

    Attributes:
        vtuber_game_graph_path: Path to the bipartite VTuber-game graph (pickle).
        collab_graph_path: Path to the VTuber collaboration graph (pickle).
        metadata_path: Path to the build metadata JSON file.
    """
    vtuber_game_graph_path: Path
    collab_graph_path: Path
    metadata_path: Path


def load_all_streams(data_dir: Path) -> list[dict]:
    """
    Load all stream records from JSON files in a directory.

    Args:
        data_dir: Directory containing JSON files with stream arrays.

    Returns:
        List of stream dictionaries extracted from all JSON files.
    """
    streams: list[dict] = []
    for fp in iter_json_files(data_dir):
        arr = load_json(fp)
        if not isinstance(arr, list):
            continue
        for s in arr:
            if isinstance(s, dict):
                streams.append(s)
    return streams


def _vtuber_node_id(channel_obj: dict) -> str:
    """
    Generate a unique node identifier for a VTuber channel.

    Prefers channel ID over name for stability, as names may change.

    Args:
        channel_obj: Dictionary with 'id' and/or 'name' fields.

    Returns:
        String identifier for the VTuber node.
    """
    cid = channel_obj.get("id")
    if cid:
        return str(cid)
    # fallback: name
    return str(channel_obj.get("name") or "unknown_channel")


def build_graphs(
    *,
    data_dir: Path,
    denylist_path: Path,
    out_dir: Path,
    cfg: WeightConfig,
) -> BuildOutputs:
    """
    Build VTuber-game bipartite and VTuber collaboration graphs from stream data.

    This function processes all stream records to construct two graphs:
    1. Bipartite graph (B): VTuber nodes connected to game/topic nodes
    2. Collaboration graph (C): VTuber nodes connected when they stream together

    Edge weights incorporate both recency decay and duration capping to
    emphasize recent, substantial engagement while preventing outliers
    from dominating.

    Args:
        data_dir: Directory containing JSON files with stream records.
        denylist_path: Path to JSON file with topic IDs to exclude.
        out_dir: Base directory for output files.
        cfg: Build configuration with weighting parameters.

    Returns:
        BuildOutputs with paths to the generated graph files and metadata.
    """
    deny = load_denylist(denylist_path)

    # Load all streams to get dataset "now" = max start_actual
    all_streams = load_all_streams(data_dir)
    tb = find_dataset_time_bounds(all_streams)
    now_utc = tb.max_dt.astimezone(timezone.utc)

    wcfg = WeightConfig(half_life_days=cfg.half_life_days, soft_cap_seconds=cfg.soft_cap_seconds)

    # Graph 1: bipartite VTuber <-> Game(topic_id)
    # We'll store VTubers as "vt:<channel_id>" and games as "g:<topic_id>" to avoid collisions.
    B = nx.Graph()
    # Graph 2: collaboration graph among VTubers
    C = nx.Graph()

    # Aggregate edge weights
    # vt-game edge: sum over streams of (recency * softcapped_duration)
    # collab edges: aggregate both (a) recency-decayed count, (b) recency-decayed duration
    vt_game_weight: dict[tuple[str, str], float] = {}
    collab_count: dict[tuple[str, str], float] = {}
    collab_dur: dict[tuple[str, str], float] = {}
    last_seen: dict[str, datetime] = {}

    # Collect VTuber channel data for enriching collab graph nodes
    # Maps vt:<channel_id> -> channel attributes dict
    vtuber_channel_data: dict[str, dict] = {}

    # Build nodes/edges
    for s in all_streams:
        topic_id = s.get("topic_id")
        if not topic_id:
            continue
        topic_id = str(topic_id)
        if topic_id in deny:
            continue

        ch = s.get("channel") or {}
        vt_id_raw = _vtuber_node_id(ch)
        vt = f"vt:{vt_id_raw}"
        g = f"g:{topic_id}"

        # Store channel data for later use in collab graph
        if vt not in vtuber_channel_data and ch:
            vtuber_channel_data[vt] = {
                "channel_id": vt_id_raw,
                "name": ch.get("name"),
                "english_name": ch.get("english_name"),
                "org": ch.get("org"),
                "suborg": ch.get("suborg"),
            }

        # timestamps
        start_actual = s.get("start_actual")
        if not start_actual:
            continue
        start_utc = parse_iso8601_z(str(start_actual))

        # duration
        dur = float(s.get("duration") or 0.0)

        # vt-game weight
        w = stream_edge_weight(now_utc=now_utc, start_utc=start_utc, duration_seconds=dur, cfg=wcfg)
        vt_game_weight[(vt, g)] = vt_game_weight.get((vt, g), 0.0) + w

        prev = last_seen.get(vt)
        if (prev is None) or (start_utc > prev):
            last_seen[vt] = start_utc


        # Store node attrs
        B.add_node(vt, bipartite="vtuber", kind="vtuber", channel_id=vt_id_raw,
                   name=ch.get("name"), english_name=ch.get("english_name"),
                   org=ch.get("org"), suborg=ch.get("suborg"))
        B.add_node(g, bipartite="game", kind="game", topic_id=topic_id)

        # collaborations
        collabs = s.get("collabs") or []
        if not isinstance(collabs, list):
            continue

        r = recency_weight(now_utc, start_utc, cfg.half_life_days)
        for cobj in collabs:
            if not isinstance(cobj, dict):
                continue
            other_id = cobj.get("id")
            if not other_id:
                continue
            other_id = str(other_id)
            other = f"vt:{other_id}"

            # Store collaborator channel data if available
            if other not in vtuber_channel_data and cobj:
                other_name = cobj.get("name")
                other_english_name = cobj.get("english_name")
                other_org = cobj.get("org")
                other_suborg = cobj.get("suborg")
                if any([other_name, other_english_name, other_org, other_suborg]):
                    vtuber_channel_data[other] = {
                        "channel_id": other_id,
                        "name": other_name,
                        "english_name": other_english_name,
                        "org": other_org,
                        "suborg": other_suborg,
                    }

            # Sym-up: treat as undirected canonical key
            a, b = sorted([vt, other])
            key = (a, b)

            # count component (recency-decayed)
            collab_count[key] = collab_count.get(key, 0.0) + r

            # duration component (recency-decayed * softcapped duration_seconds)
            cdur = float(cobj.get("duration_seconds") or 0.0)
            cdur_eff = softcap_duration(cdur, cfg.soft_cap_seconds)
            collab_dur[key] = collab_dur.get(key, 0.0) + r * cdur_eff

    # Materialize bipartite edges
    for (vt, g), w in vt_game_weight.items():
        B.add_edge(vt, g, weight=w)

    # Materialize collab edges with both components
    # Provide a combined weight too (for diffusion), defaulting to normalized mix later.
    for (a, b), cnt in collab_count.items():
        dur = collab_dur.get((a, b), 0.0)
        C.add_edge(a, b, count_weight=cnt, duration_weight=dur)

    # Enrich collab graph nodes with full attributes (same as bipartite graph)
    for vt_node, attrs in vtuber_channel_data.items():
        C.add_node(
            vt_node,
            kind="vtuber",
            channel_id=attrs.get("channel_id"),
            name=attrs.get("name"),
            english_name=attrs.get("english_name"),
            org=attrs.get("org"),
            suborg=attrs.get("suborg"),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "artifacts"
    meta_dir.mkdir(parents=True, exist_ok=True)

    vg_path = graphs_dir / "vtuber_game.pickle"
    co_path = graphs_dir / "collab.pickle"
    md_path = meta_dir / "build_metadata.json"

    with open(vg_path, "wb") as f:
        pickle.dump(B, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(co_path, "wb") as f:
        pickle.dump(C, f, protocol=pickle.HIGHEST_PROTOCOL)

    last_seen_path = meta_dir / "last_seen.json"
    save_json(last_seen_path, {vt: dt.isoformat() for vt, dt in last_seen.items()})
    save_json(md_path, {
        "now_utc": now_utc.isoformat(),
        "min_utc": tb.min_dt.isoformat(),
        "max_utc": tb.max_dt.isoformat(),
        "half_life_days": cfg.half_life_days,
        "soft_cap_seconds": cfg.soft_cap_seconds,
        "denylist_size": len(deny),
        "vtuber_game_edges": B.number_of_edges(),
        "vtuber_nodes": sum(1 for n, d in B.nodes(data=True) if d.get("kind") == "vtuber"),
        "game_nodes": sum(1 for n, d in B.nodes(data=True) if d.get("kind") == "game"),
        "collab_edges": C.number_of_edges(),
        "collab_nodes": C.number_of_nodes(),
    })

    return BuildOutputs(
        vtuber_game_graph_path=vg_path,
        collab_graph_path=co_path,
        metadata_path=md_path,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the graph builder.

    Returns:
        Namespace containing parsed arguments with defaults applied.
    """
    parser = argparse.ArgumentParser(
        description="Build VTuber collaboration and game interaction graphs from stream data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing JSON files with stream data",
    )
    parser.add_argument(
        "--denylist",
        type=Path,
        default=Path("denylist.json"),
        help="Path to JSON file with topic IDs to exclude",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=180.0,
        help="Recency weight half-life in days",
    )
    parser.add_argument(
        "--soft-cap-seconds",
        type=float,
        default=18000.0,
        help="Maximum duration for weight calculation (5 hours = 18000 seconds)",
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the graph builder CLI.

    Parses arguments, builds graphs from stream data, and saves outputs.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    args = parse_args()

    if not args.data_dir.exists():
        print(f"Error: Data directory does not exist: {args.data_dir}", file=sys.stderr)
        return 1

    cfg = WeightConfig(
        half_life_days=args.half_life_days,
        soft_cap_seconds=args.soft_cap_seconds,
    )

    try:
        outputs = build_graphs(
            data_dir=args.data_dir,
            denylist_path=args.denylist,
            out_dir=args.out_dir,
            cfg=cfg,
        )
        print(f"Built graphs successfully!")
        print(f"  VTuber-Game graph: {outputs.vtuber_game_graph_path}")
        print(f"  Collab graph: {outputs.collab_graph_path}")
        print(f"  Metadata: {outputs.metadata_path}")
        return 0
    except Exception as e:
        print(f"Error building graphs: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
