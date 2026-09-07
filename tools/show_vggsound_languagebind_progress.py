#!/usr/bin/env python3
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs/languagebind_features/VGGSound/vggsound_main_segments.npz"
STAGING = Path("/home/wwj/文档/AVGZSL/rawVGGSound/.languagebind_staging")
TOTAL = 93752


def main() -> None:
    print(f"time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if CACHE.is_file():
        with np.load(CACHE, allow_pickle=False) as cache:
            encoded = len(cache["video_names"])
            failures = len(json.loads(str(cache["failures_json"])))
        updated = datetime.fromtimestamp(CACHE.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"saved={encoded}/{TOTAL} ({encoded / TOTAL:.2%}) "
            f"failures={failures} cache_updated={updated}"
        )
    else:
        print(f"cache_missing={CACHE}")

    staged = list(STAGING.glob("*")) if STAGING.is_dir() else []
    if staged:
        shard = staged[0]
        video_count = sum(1 for _ in shard.glob("*.mp4"))
        print(f"staging_shard={shard.name} staged_videos={video_count}")
    else:
        print("staging_shard=none")

    processes = subprocess.run(
        ["pgrep", "-af", "extract_vggsound_languagebind_shards.py|extract_languagebind_vggsound.py"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print("process=running" if processes else "process=stopped")

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if gpu:
        print(f"gpu={gpu}")


if __name__ == "__main__":
    main()
