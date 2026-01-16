#!/usr/bin/env python3
"""
Standalone Holodex Client + Collaborations Extractor

- Fully self-contained (no external project imports).
- Talks to the Holodex API with:
    - API key header
    - User-Agent
    - Local rate-limiting (80 req / 120s)
    - 429 retry & network retry
- Fetches all VTuber channels.
- For each VTuber, fetches all past streams with enriched metadata:
    include=mentions,live_info  (required for collab info)
- For each stream, computes collabs and collab durations.
- Saves ONE JSON file per VTuber, containing a list of streams:
    vtuber_streams_with_collabs/<sanitized_name>_<channel_id>.json

Each stream object looks like:
{
  ... (all Holodex fields) ...,
  "collabs": [
    { "id": "...", "name": "...", "duration_seconds": 1234 },
    ...
  ]
}
"""

import os
import re
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List

from datetime import datetime

import requests


# ============================================================
#  CONFIG
# ============================================================

HOLODEX_BASE_URL = "https://holodex.net/api/v2"
HOLODEX_API_KEY_ENV = "HOLODEX_API_KEY"

# Directory to store per-VTuber JSON files
OUTPUT_DIR = Path("vtuber_streams_with_collabs")

# Holodex rate limit: 80 requests / 120 seconds
RATE_WINDOW_SECONDS = 120
RATE_MAX_REQUESTS = 80


# ============================================================
#  UTILS
# ============================================================

def safe_filename(name: str) -> str:
    """
    Make a VTuber name safe for use as a filename.
    - Replace non-alphanumerics with underscores.
    - Collapse multiple underscores.
    - Strip leading/trailing underscores.
    """
    name = re.sub(r"[^0-9A-Za-zぁ-んァ-ン一-龯]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "vtuber"


def compute_duration_seconds(stream: Dict[str, Any]) -> Optional[int]:
    """
    Compute the duration of a stream in seconds.

    Priority:
      1. start_actual / end_actual (if both present and parseable)
      2. Holodex's 'duration' field (already in seconds)
      3. None if all else fails
    """
    start = stream.get("start_actual")
    end = stream.get("end_actual")

    if start and end:
        try:
            s = datetime.fromisoformat(start.replace("Z", "+00:00"))
            e = datetime.fromisoformat(end.replace("Z", "+00:00"))
            return max(int((e - s).total_seconds()), 0)
        except Exception:
            # fall back to 'duration' below
            pass

    # Fallback to Holodex's own duration, if present
    duration = stream.get("duration")
    if isinstance(duration, (int, float)) and duration >= 0:
        return int(duration)

    return None


# ============================================================
#  HOLODEX CLIENT (STANDALONE)
# ============================================================

class HolodexClient:
    """
    Fully standalone Holodex API client with:
    - User-Agent header
    - Automatic pagination
    - Local rate-limiting (80 req / 120s)
    - 429 / network retry
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv(HOLODEX_API_KEY_ENV)
        if not self.api_key:
            raise ValueError(
                f"[Holodex] API key not found. Set {HOLODEX_API_KEY_ENV} environment variable."
            )

        self.session = requests.Session()
        self.session.headers.update({
            "X-APIKEY": self.api_key,
            "User-Agent": "standalone-holodex-collab-client/1.0"
        })

        self.rate_window = RATE_WINDOW_SECONDS
        self.max_requests = RATE_MAX_REQUESTS
        self.request_timestamps: List[float] = []

    # -----------------------------------------------------
    # RATE LIMITING
    # -----------------------------------------------------
    def _rate_limit_wait(self):
        now = time.time()
        self.request_timestamps = [
            t for t in self.request_timestamps
            if now - t < self.rate_window
        ]

        if len(self.request_timestamps) >= self.max_requests:
            oldest = self.request_timestamps[0]
            wait_time = self.rate_window - (now - oldest)
            wait_time = max(wait_time, 1.0)
            print(f"[Holodex] Local rate limit reached. Waiting {wait_time:.1f}s…")
            time.sleep(wait_time)

    # -----------------------------------------------------
    # LOW-LEVEL REQUEST
    # -----------------------------------------------------
    def _request(self, method: str, path: str, params: Dict[str, Any]):
        url = f"{HOLODEX_BASE_URL}{path}"

        while True:
            self._rate_limit_wait()

            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=10,
                )
            except Exception as e:
                print(f"[Holodex] Network error: {e}. Retrying in 5s…")
                time.sleep(5)
                continue

            self.request_timestamps.append(time.time())

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else 5.0
                print(f"[Holodex] 429 Too Many Requests. Waiting {wait:.1f}s…")
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"[Holodex] Error {resp.status_code}\n"
                f"URL: {resp.url}\n"
                f"Response: {resp.text[:300]}"
            )

    def _get(self, path: str, params: Dict[str, Any]):
        return self._request("GET", path, params)

    # -----------------------------------------------------
    # VTUBER LISTING
    # -----------------------------------------------------
    def get_vtuber_channels(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all VTuber channels, automatically paginated.

        NOTE: This returns EVERY vtuber Holodex knows about, regardless of org.
              You can filter manually (e.g., org contains "holo") if desired.
        """
        all_channels: List[Dict[str, Any]] = []
        offset = 0

        while True:
            params = {
                "type": "vtuber",
                "limit": min(limit, 100),
                "offset": offset,
            }
            batch = self._get("/channels", params=params)
            if not batch:
                break

            all_channels.extend(batch)

            offset += limit
            if len(batch) < limit:
                break

        return all_channels

    # -----------------------------------------------------
    # STREAM LISTING WITH ENRICHED FIELDS
    # -----------------------------------------------------
    def get_channel_streams_page(
        self,
        channel_id: str,
        limit: int = 50,
        offset: int = 0,
        type_: str = "stream",
    ) -> List[Dict[str, Any]]:
        """
        Get one page of streams for a channel.

        IMPORTANT:
        - include=mentions,live_info → enables collab fields
        - status=past → ensures start_actual / end_actual where possible
        """
        params = {
            "channel_id": channel_id,
            "type": type_,
            "limit": min(limit, 100),
            "offset": offset,
            "include": "mentions,live_info",
            "status": "past",
        }
        return self._get("/videos", params=params)

    # -----------------------------------------------------
    # STREAM ITERATOR
    # -----------------------------------------------------
    def iter_channel_streams(
        self,
        channel_id: str,
        per_page: int = 50,
        max_pages: Optional[int] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        Yield all streams for a VTuber channel, automatically paginated.
        """
        offset = 0
        pages = 0

        while True:
            data = self.get_channel_streams_page(
                channel_id=channel_id,
                limit=per_page,
                offset=offset,
            )
            if not data:
                break

            for item in data:
                yield item

            offset += per_page
            pages += 1

            if max_pages is not None and pages >= max_pages:
                break

    # -----------------------------------------------------
    # COLLAB EXTRACTION
    # -----------------------------------------------------
    @staticmethod
    def extract_collabs(
        stream: Dict[str, Any],
        main_id: str,
        main_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract collaboration info from a single stream.

        - Collects:
            - main vtuber
            - mentions (type=vtuber)
            - live_info.collaborators
        - Computes duration_seconds for the stream.
        - Returns a list of collaborators (excluding the main vtuber), each:
            { "id": str, "name": str, "duration_seconds": int | None }
        """

        duration = compute_duration_seconds(stream)

        participants: Dict[str, Dict[str, Any]] = {
            main_id: {"id": main_id, "name": main_name}
        }

        # mentions
        for m in stream.get("mentions", []) or []:
            if m.get("type") == "vtuber" and m.get("id"):
                participants[m["id"]] = {"id": m["id"], "name": m.get("name", "")}

        # live_info.collaborators
        live_info = stream.get("live_info") or {}
        for c in live_info.get("collaborators", []) or []:
            if c.get("id"):
                participants[c["id"]] = {"id": c["id"], "name": c.get("name", "")}

        collabs: List[Dict[str, Any]] = []
        for pid, info in participants.items():
            if pid == main_id:
                continue
            collabs.append({
                "id": pid,
                "name": info["name"],
                "duration_seconds": duration,
            })

        return collabs

    # -----------------------------------------------------
    # SAVE ONE VTUBER'S STREAMS + COLLABS TO FILE
    # -----------------------------------------------------
    def save_vtuber_streams_with_collabs(
        self,
        vtuber_id: str,
        vtuber_name: str,
        out_dir: Path = OUTPUT_DIR,
        per_page: int = 50,
    ) -> Path:
        """
        Fetch all past streams for a VTuber, enrich with 'collabs', and save to a JSON file.

        Returns the Path to the JSON file.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = safe_filename(vtuber_name)
        out_path = out_dir / f"{base_name}_{vtuber_id}.json"

        streams: List[Dict[str, Any]] = []
        stream_count = 0
        collab_stream_count = 0

        for stream in self.iter_channel_streams(
            channel_id=vtuber_id,
            per_page=per_page,
        ):
            stream_count += 1
            collabs = self.extract_collabs(stream, vtuber_id, vtuber_name)
            if collabs:
                collab_stream_count += 1
            stream["collabs"] = collabs
            streams.append(stream)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(streams, f, ensure_ascii=False, indent=2)

        print(
            f"[Holodex] Saved {stream_count} streams "
            f"({collab_stream_count} with collabs) → {out_path}"
        )

        return out_path


# ============================================================
#  MAIN ENTRY POINT
# ============================================================

def main():
    client = HolodexClient()

    print("[Holodex] Fetching VTuber channels…")
    vtubers = client.get_vtuber_channels()
    print(f"[Holodex] Total VTubers returned: {len(vtubers)}")

    # restrict to Hololive-related channels
    vtubers = [
        v for v in vtubers
        if "holo" in (v.get("org") or "").lower()
    ]
    print(f"[Holodex] After Hololive-org filter: {len(vtubers)}")

    for v in vtubers:
        cid = v.get("id")
        name = v.get("name", cid)
        if not cid:
            continue

        print(f"\n[Holodex] Processing VTuber: {name} ({cid})")
        try:
            client.save_vtuber_streams_with_collabs(
                vtuber_id=cid,
                vtuber_name=name,
                out_dir=OUTPUT_DIR,
                per_page=50,
            )
        except Exception as e:
            print(f"[Holodex] Error while processing {name} ({cid}): {e}")


if __name__ == "__main__":
    main()
