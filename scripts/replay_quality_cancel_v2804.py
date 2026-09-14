#!/usr/bin/env python3
"""Write a separate historical cancel interpretation; never edit the source."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onnx_splitpoint_tool.quality_lifecycle import replay_historical_cancellation
from onnx_splitpoint_tool.workflow.scientific_reporting import project_central_quality_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    evidence = json.loads(args.evidence.read_text())
    replay = replay_historical_cancellation(summary, evidence)
    report = project_central_quality_status(replay)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "central_quality_cancel_replay.json").write_text(json.dumps(replay, indent=2) + "\n")
    (args.output_dir / "quality_status_projection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"QUALITY_CANCEL_REPLAY={args.output_dir}")


if __name__ == "__main__":
    main()
