#!/usr/bin/env python3
"""
Clean Holodex-like stream JSON files.

For each JSON file in INPUT_DIR:
- Load JSON (expects a list of entries OR a dict with a list under common keys).
- Keep only entries where:
    - topic_id exists and is non-empty
    - status == "past"
- For each kept entry, retain ONLY:
    id, title, topic_id, start_actual, end_actual, duration,
    channel: {id, name, english_name, org, suborg},
    collabs: [{id, name, duration_seconds}]
- Save cleaned JSON to OUTPUT_DIR with same filename.

Usage:
  python clean_stream_json.py --input /path/to/in --output /path/to/out
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _as_list(payload: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Accept either:
      - list[dict]
      - dict with list under one of these keys: "videos", "streams", "data", "items"
    Returns (list, key_used) where key_used is None if payload itself is a list.
    """
    if isinstance(payload, list):
        return payload, None

    if isinstance(payload, dict):
        for key in ("videos", "streams", "data", "items"):
            v = payload.get(key)
            if isinstance(v, list):
                # filter only dict entries; ignore junk
                return [x for x in v if isinstance(x, dict)], key

    raise ValueError("Unsupported JSON structure (expected list or dict containing a list).")


def _clean_channel(ch: Any) -> Dict[str, Any]:
    if not isinstance(ch, dict):
        return {}
    keep = ("id", "name", "english_name", "org", "suborg")
    return {k: ch.get(k) for k in keep if k in ch}


def _clean_collabs(collabs: Any) -> List[Dict[str, Any]]:
    if not isinstance(collabs, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for c in collabs:
        if not isinstance(c, dict):
            continue
        item = {}
        for k in ("id", "name", "duration_seconds"):
            if k in c:
                item[k] = c.get(k)
        if item:  # keep non-empty
            cleaned.append(item)
    return cleaned


def _is_nonempty_topic_id(entry: Dict[str, Any]) -> bool:
    if "topic_id" not in entry:
        return False
    tid = entry.get("topic_id")
    if tid is None:
        return False
    if isinstance(tid, str) and tid.strip() == "":
        return False
    return True


def clean_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Filters
    if not _is_nonempty_topic_id(entry):
        return None
    if entry.get("status") != "past":
        return None

    # Keep only requested fields
    out: Dict[str, Any] = {}
    for k in ("id", "title", "topic_id", "start_actual", "end_actual", "duration"):
        if k in entry:
            out[k] = entry.get(k)

    out["channel"] = _clean_channel(entry.get("channel"))
    out["collabs"] = _clean_collabs(entry.get("collabs"))

    return out


def clean_file(in_path: Path, out_path: Path, indent: int = 2) -> Dict[str, Any]:
    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    entries, key_used = _as_list(payload)

    kept: List[Dict[str, Any]] = []
    removed_no_topic = 0
    removed_not_past = 0

    for e in entries:
        if not _is_nonempty_topic_id(e):
            removed_no_topic += 1
            continue
        if e.get("status") != "past":
            removed_not_past += 1
            continue

        cleaned = clean_entry(e)
        if cleaned is not None:
            kept.append(cleaned)

    # Preserve original outer structure if it was dict-with-list
    if key_used is None:
        out_payload: Union[List[Dict[str, Any]], Dict[str, Any]] = kept
    else:
        # copy dict but replace only the list portion
        if not isinstance(payload, dict):
            # should not happen, but keep safe
            out_payload = {key_used: kept}
        else:
            out_payload = dict(payload)
            out_payload[key_used] = kept

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=indent)

    return {
        "file": str(in_path.name),
        "total_in": len(entries),
        "kept": len(kept),
        "removed_no_topic_id": removed_no_topic,
        "removed_not_past": removed_not_past,
        "structure": "list" if key_used is None else f"dict[{key_used}]",
    }


def iter_json_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.json") if p.is_file()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input directory containing JSON files")
    ap.add_argument("--output", required=True, help="Output directory to write cleaned JSON files")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent level (default: 2)")
    ap.add_argument("--preserve-subdirs", action="store_true",
                    help="Preserve subdirectory structure under output directory")
    args = ap.parse_args()

    in_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {in_dir}")

    files = iter_json_files(in_dir)
    if not files:
        raise SystemExit(f"No .json files found under: {in_dir}")

    stats = []
    for in_path in files:
        if args.preserve_subdirs:
            rel = in_path.relative_to(in_dir)
            out_path = out_dir / rel
        else:
            out_path = out_dir / in_path.name

        try:
            s = clean_file(in_path, out_path, indent=args.indent)
            stats.append(s)
            print(f"[OK] {in_path.name}: kept {s['kept']}/{s['total_in']} ({s['structure']})")
        except Exception as e:
            print(f"[FAIL] {in_path}: {e}")

    total_in = sum(s["total_in"] for s in stats)
    total_kept = sum(s["kept"] for s in stats)
    total_no_topic = sum(s["removed_no_topic_id"] for s in stats)
    total_not_past = sum(s["removed_not_past"] for s in stats)

    print("\n=== Summary ===")
    print(f"Files processed: {len(stats)}/{len(files)}")
    print(f"Entries in:      {total_in}")
    print(f"Entries kept:    {total_kept}")
    print(f"Removed (no topic_id): {total_no_topic}")
    print(f"Removed (status != past): {total_not_past}")
    print(f"Output dir:      {out_dir}")


if __name__ == "__main__":
    main()
