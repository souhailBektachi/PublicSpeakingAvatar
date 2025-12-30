#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Iterator, Dict
import sys

# Ensure 'backend/src' is importable when running the script directly
try:
    from src.services.analyzers.pose_interpreter import PoseFeedbackEngine
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.services.analyzers.pose_interpreter import PoseFeedbackEngine

BACKEND_DIR = Path(__file__).resolve().parents[1]
POSE_IN_FILE = BACKEND_DIR / "assets" / "pose" / "pose_telemetry.jsonl"
POSE_OUT_FILE = BACKEND_DIR / "assets" / "pose" / "pose_interpretation.jsonl"


def iter_jsonl_tail(path: Path, poll_interval: float = 0.2) -> Iterator[Dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    with path.open("r", encoding="utf-8") as f:
        # Seek to end by default
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_interval)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue


def iter_jsonl_start(path: Path, poll_interval: float = 0.2) -> Iterator[Dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    with path.open("r", encoding="utf-8") as f:
        # Read existing lines, then follow new
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                time.sleep(poll_interval)
                f.seek(pos)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    parser = argparse.ArgumentParser(description="Realtime pose interpreter for JSONL telemetry")
    parser.add_argument("--from-start", action="store_true", help="Process existing lines then follow new ones")
    parser.add_argument("--in", dest="in_path", default=str(POSE_IN_FILE), help="Input telemetry JSONL file")
    parser.add_argument("--out", dest="out_path", default=str(POSE_OUT_FILE), help="Output interpretation JSONL file")
    parser.add_argument("--poll", dest="poll", type=float, default=0.2, help="Polling interval in seconds")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    engine = PoseFeedbackEngine()
    iterator = iter_jsonl_start(in_path, args.poll) if args.from_start else iter_jsonl_tail(in_path, args.poll)

    with out_path.open("a", encoding="utf-8") as out:
        for chunk in iterator:
            result = engine.process_chunk(chunk)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
