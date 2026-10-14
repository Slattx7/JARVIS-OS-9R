import json, tempfile
from pathlib import Path

from jarvis_loop import Outcome, Trace


def T(i, task, tools=("a",), ok=True, viol=0, **kw):
    d = {"id": i, "task": task, "steps": [{"tool": t, "args": {}} for t in tools], "success": ok, "violations": viol}
    d.update(kw)
    return Trace.from_dict(d)


def O(oks, viol=None, lat=100.0, err=None):
    """ids are str(index); oks may contain None (no gold)."""
    return {str(i): Outcome(o, (viol or {}).get(i, 0), lat, (err or {}).get(i, "")) for i, o in enumerate(oks)}


def tmpdir():
    d = tempfile.TemporaryDirectory()
    return d, Path(d.name)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
