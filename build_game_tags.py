"""
build_game_tags.py - Generate Game Tags Template from Stream Data

This module scans stream data to extract all unique topic_ids and generates
a template JSON file for tagging games with categories like "fps", "horror",
"cozy", etc.

The output format maps each topic_id to an entry containing:
- inputs: List of identifiers that map to this entry (initially just the topic_id)
- is_game: Boolean flag (True by default; set False for non-game topics)
- tags: List of category tags to be filled in manually

Usage:
    # Generate a template with all topics except those in denylist
    python build_game_tags.py --data-dir streams/ --denylist denylist.txt --output game_tags.json

Workflow:
    1. Run this script to generate game_tags.json with all topics
    2. Edit game_tags.json to:
       - Add tags like "fps", "horror", "vtuber_game", etc.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


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
    Load a denylist of topic IDs to exclude from processing.

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


@dataclass(frozen=True)
class GameTagsEntry:
    """
    Schema for a single entry in the game tags file.

    Attributes:
        inputs: List of identifiers that resolve to this entry (topic_ids, aliases).
        is_game: Whether this entry represents a game (False for talks, events, etc.).
        tags: List of category tags for filtering/analysis (e.g., ["fps", "horror"]).
    """
    inputs: list[str]
    is_game: bool
    tags: list[str]


def collect_topic_ids(data_dir: Path) -> set[str]:
    """
    Scan all JSON files in a directory and collect unique topic_ids.

    Args:
        data_dir: Directory containing JSON files with stream records.

    Returns:
        Set of unique topic_id strings found across all files.
    """
    topics: set[str] = set()
    for fp in iter_json_files(data_dir):
        arr = load_json(fp)
        if not isinstance(arr, list):
            continue
        for s in arr:
            tid = s.get("topic_id")
            if tid:
                topics.add(str(tid))
    return topics


def generate_blank_game_tags(
    *,
    data_dir: Path,
    denylist_path: Path,
    out_path: Path,
    include_denied: bool = False,
) -> None:
    """
    Generate a template game_tags.json file with all discovered topics.

    Creates a mapping of topic_id -> entry structure ready for manual tagging.
    By default, topics in the denylist are excluded from the output.

    Args:
        data_dir: Directory containing JSON files with stream data.
        denylist_path: Path to JSON file with topic IDs to exclude.
        out_path: Path for the output game_tags.json file.
        include_denied: If True, include denylisted topics in output (default: False).
    """
    deny = load_denylist(denylist_path)
    topics = collect_topic_ids(data_dir)
    topics_sorted = sorted(topics)

    obj: dict[str, dict] = {}
    for tid in topics_sorted:
        if (tid in deny) and (not include_denied):
            continue
        obj[tid] = {
            "inputs": [tid],
            # Mark as game by default; your denylist is where you exclude non-game topics
            "is_game": True,
            "tags": [],
        }

    save_json(out_path, obj)
    print(f"Generated {out_path} with {len(obj)} topics")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the game tags generator.

    Returns:
        Namespace containing parsed arguments with defaults applied.
    """
    parser = argparse.ArgumentParser(
        description="Generate a game tags template from stream data",
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
        "--output",
        type=Path,
        default=Path("game_tags.json"),
        help="Path for output game_tags.json file",
    )
    parser.add_argument(
        "--include-denied",
        action="store_true",
        help="Include denylisted topics in output (not recommended)",
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the game tags generator CLI.

    Parses arguments and generates the game_tags.json template.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    args = parse_args()

    if not args.data_dir.exists():
        print(f"Error: Data directory does not exist: {args.data_dir}", file=sys.stderr)
        return 1

    try:
        generate_blank_game_tags(
            data_dir=args.data_dir,
            denylist_path=args.denylist,
            out_path=args.output,
            include_denied=args.include_denied,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
