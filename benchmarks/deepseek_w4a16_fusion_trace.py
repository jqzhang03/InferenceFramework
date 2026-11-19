"""Count expert-region and whole-model launches in the fusion experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from benchmarks.cuda_operator_profile_report import union_ns


def analyze(path: Path) -> dict:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        kernels = defaultdict(list)
        for pid, correlation, start, end, name in db.execute("""
            SELECT k.globalPid, k.correlationId, k.start, k.end, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        """):
            kernels[pid, correlation].append((start, end, name))
        labels = list(db.execute("""
            SELECT text, start, end, globalTid FROM NVTX_EVENTS
            WHERE end IS NOT NULL AND (text LIKE 'moe_fusion_%' OR text LIKE 'step/%')
            ORDER BY start
        """))
        experts, model = {}, {}
        for label, begin, end, tid in labels:
            kind, *parts = label.split("/")
            if kind == "moe_fusion_model":
                mode, = parts
                model[mode] = dict(steps=[], start=begin, end=end)
                continue
            calls = db.execute("""
                SELECT DISTINCT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
                WHERE globalTid=? AND start>=? AND end<=?
            """, (tid, begin, end))
            events = [event for (correlation,) in calls
                      for event in kernels.get((tid & ~0xFFFFFF, correlation), ())]
            counts = Counter(name for _, _, name in events)
            row = dict(launches=len(events), counts=counts, wall_ms=(end - begin) / 1e6,
                       active_ms=union_ns([(start, stop) for start, stop, _ in events]) / 1e6)
            if kind == "moe_fusion_expert":
                mode, m = parts
                experts.setdefault(mode, {})[m] = row
            elif begin >= model[mode]["start"] and end <= model[mode]["end"]:
                model[mode]["steps"].append(row)
    for mode, expected in (("none", 5), ("gate_up", 2), ("full", 1)):
        if len(experts.get(mode, {})) != 4 or any(
                row["launches"] != expected for row in experts[mode].values()):
            raise ValueError(f"unexpected traced launches in isolated {mode}: {experts.get(mode)}")
        if len(model.get(mode, {}).get("steps", [])) != 4:
            raise ValueError(f"missing model trace for {mode}")
    return dict(experts=experts, model=model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.sqlite), indent=2))


if __name__ == "__main__":
    main()
