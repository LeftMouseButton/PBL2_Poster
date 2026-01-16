"""
inspect.py - Inspect and Analyze VTuber Graph Data

This module provides CLI tools for inspecting and analyzing the graphs built by
build_graphs.py. It can:
1. List top games/topics by total engagement weight
2. Show detailed breakdowns for specific VTubers with game-level metrics

The VTuber-game graph (bipartite) connects VTuber nodes to game/topic nodes,
with edge weights representing recency-decayed engagement. This module
provides utilities to explore these weights and filter by game tags (e.g.,
identifying FPS games).

Usage:
    # List top 30 topics by total weight
    python inspect.py top-topics --graph graphs/graphs/vtuber_game.pickle --tags game_tags.json

    # List top games for a specific VTuber by weight
    python inspect.py vtuber --graph graphs/graphs/vtuber_game.pickle --tags game_tags.json --name "Towa"

    //# List top FPS games for a VTuber
    //python inspect.py vtuber --graph graphs/graphs/vtuber_game.pickle --tags game_tags.json --name "Towa" --tag fps

Output formats:
    - Tables with rankings, IDs, weights, and tag indicators
    - Summary statistics (total weight, filtered weight, fraction)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(fp: Path) -> Any:
    """
    Load and parse a JSON file.

    Args:
        fp: Path to the JSON file.

    Returns:
        The parsed JSON content (dict, list, or primitive type).
    """
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_vtuber_fps_breakdown(
    vtuber_game_pickle: Path,
    game_tags_path: Path,
    vtuber_contains: str,
    tag: str = "fps",
    top_n_games: int = 25,
) -> None:
    """
    Find VTuber nodes matching a search term and show their top games with tag filtering.

    Searches for VTuber nodes by matching against name, english_name, or channel_id.
    Displays their top games by edge weight, indicating which have the specified tag.

    Args:
        vtuber_game_pickle: Path to the bipartite VTuber-game graph pickle file.
        game_tags_path: Path to JSON file mapping topic_id -> tags list.
        vtuber_contains: Case-insensitive search term for VTuber name/ID.
        tag: Game tag to filter/flag (default: "fps" for FPS games).
        top_n_games: Maximum number of games to display per VTuber (default: 25).
    """
    with open(vtuber_game_pickle, "rb") as f:
        B = pickle.load(f)

    game_tags = load_json(game_tags_path)

    q = vtuber_contains.lower()

    # find matching vtubers by searching across name fields
    matches = []
    for n, d in B.nodes(data=True):
        if not (isinstance(n, str) and n.startswith("vt:")):
            continue
        name = (d.get("name") or "")
        en = (d.get("english_name") or "")
        cid = (d.get("channel_id") or n[3:])
        hay = f"{name} {en} {cid}".lower()
        if q in hay:
            matches.append(n)

    if not matches:
        print(f"No vtuber matched: {vtuber_contains}")
        return

    for vt in matches:
        d = B.nodes[vt]
        label = d.get("english_name") or d.get("name") or vt[3:]
        print("\n====================================================")
        print(f"VTuber: {label}   ({vt})")
        print("====================================================")

        # collect games with weights and tag status
        rows = []
        total = 0.0
        fps_total = 0.0

        for nb in B.neighbors(vt):
            if not (isinstance(nb, str) and nb.startswith("g:")):
                continue
            ed = B.get_edge_data(vt, nb) or {}
            w = float(ed.get("weight") or 0.0)
            if w <= 0:
                continue

            topic_id = nb[2:]
            tags = game_tags.get(topic_id, {}).get("tags", [])
            has = tag in tags

            total += w
            if has:
                fps_total += w

            rows.append((topic_id, w, has, tags))

        rows.sort(key=lambda x: x[1], reverse=True)

        print(f"Total weight (all games): {total:.1f}")
        print(f"Total weight ({tag} games): {fps_total:.1f}")
        print(f"Fraction: {(fps_total/total if total>0 else 0.0):.3f}")
        print()
        print("Rank | topic_id                       | weight       | has_tag")
        print("-----+--------------------------------+--------------+--------")

        for i, (tid, w, has, tags) in enumerate(rows[:top_n_games], 1):
            print(f"{i:>4} | {tid:<30} | {w:>12.1f} | {has}")


def list_top_topics(
    vtuber_game_pickle: Path,
    game_tags_path: Path,
    top_n: int = 30,
    tag: str = "fps",
) -> None:
    """
    List the top topics/games by total engagement weight across all VTubers.

    Aggregates edge weights from the bipartite graph to rank topics by
    their overall popularity within the VTuber streaming ecosystem.

    Args:
        vtuber_game_pickle: Path to the bipartite VTuber-game graph pickle file.
        game_tags_path: Path to JSON file mapping topic_id -> tags list.
        top_n: Number of top topics to display (default: 30).
        tag: Game tag to indicate in output (default: "fps").
    """
    # load graph directly (not using networkx helper)
    with open(vtuber_game_pickle, "rb") as f:
        B = pickle.load(f)

    game_tags = load_json(game_tags_path)

    topic_weight = defaultdict(float)

    # aggregate weights across all VTuber-game edges
    for u, v, ed in B.edges(data=True):
        g = u if isinstance(u, str) and u.startswith("g:") else v
        if not (isinstance(g, str) and g.startswith("g:")):
            continue
        topic_weight[g[2:]] += float(ed.get("weight") or 0.0)

    ranked = sorted(topic_weight.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\nTop {top_n} topic_ids by total weight:\n")
    print("Rank | topic_id                       | total_weight | has_tag")
    print("-----+--------------------------------+--------------+--------")

    for i, (tid, w) in enumerate(ranked, 1):
        tags = game_tags.get(tid, {}).get("tags", [])
        print(f"{i:>4} | {tid:<30} | {w:>12.1f} | {tag in tags}")


def create_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """
    Add subcommands to an argparse parser for the inspection utilities.

    Registers 'top-topics' and 'vtuber' subcommands with their respective
    argument configurations.

    Args:
        subparsers: The subparsers action from argparse.add_subparsers().
    """
    parser_topics = subparsers.add_parser(
        "top-topics",
        help="List top games/topics by total engagement weight",
        description="Aggregates edge weights from the VTuber-game graph to rank "
                   "topics by overall popularity.",
    )
    parser_topics.add_argument(
        "--graph",
        type=Path,
        required=True,
        help="Path to the vtuber_game.gpickle file",
    )
    parser_topics.add_argument(
        "--tags",
        type=Path,
        required=True,
        help="Path to game_tags.json file",
    )
    parser_topics.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top topics to display (default: 30)",
    )
    parser_topics.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag to check for in output (default: none)",
    )
    parser_topics.set_defaults(func=list_top_topics)

    parser_vtuber = subparsers.add_parser(
        "vtuber",
        help="Show detailed breakdown for a specific VTuber",
        description="Finds VTuber nodes by name/ID and displays their top games "
                   "with optional tag filtering.",
    )
    parser_vtuber.add_argument(
        "--graph",
        type=Path,
        required=True,
        help="Path to the vtuber_game.pickle file",
    )
    parser_vtuber.add_argument(
        "--tags",
        type=Path,
        required=True,
        help="Path to game_tags.json file",
    )
    parser_vtuber.add_argument(
        "--name",
        type=str,
        required=True,
        help="Search term for VTuber name (case-insensitive partial match)",
    )
    parser_vtuber.add_argument(
        "--tag",
        type=str,
        default="",
        help="Game tag to filter/flag (default: none)",
    )
    parser_vtuber.add_argument(
        "--top_n_games",
        type=int,
        default=25,
        help="Number of games to display per VTuber (default: 25)",
    )
    parser_vtuber.set_defaults(func=inspect_vtuber_fps_breakdown)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the graph inspector.

    Supports two subcommands:
        - top-topics: List top games by total weight
        - vtuber: Show detailed breakdown for a specific VTuber

    Returns:
        Namespace containing parsed arguments with subcommand function.
    """
    parser = argparse.ArgumentParser(
        description="Inspect and analyze VTuber graph data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Inspection mode")
    create_subparsers(subparsers)
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the graph inspector CLI.

    Parses arguments and dispatches to the appropriate inspection function.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    args = parse_args()

    if args.command is None:
        sys.stderr.write("Error: No subcommand specified. Use --help for usage.\n")
        return 1

    try:
        if args.command == "top-topics":
            args.func(
                vtuber_game_pickle=args.graph,
                game_tags_path=args.tags,
                top_n=args.top_n,
                tag=args.tag,
            )
        elif args.command == "vtuber":
            args.func(
                vtuber_game_pickle=args.graph,
                game_tags_path=args.tags,
                vtuber_contains=args.name,
                tag=args.tag,
                top_n_games=args.top_n_games,
            )
        else:
            raise ValueError(f"Unknown subcommand: {args.command}")
        return 0
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
