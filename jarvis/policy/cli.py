from __future__ import annotations
import argparse, json, os, sys

from .backends.frida_be import build_manifest
from .config import load_config
from .confirm import CLIConfirmer
from .core import Action, Audit, Source
from .planner import PlanError, parse_plan, run_plan
from .registry import build_dispatcher, build_registry
from .runner import Runner
from .trace import replay


def _coerce(schema, kvs):
    out = {}
    for kv in kvs or []:
        k, _, v = kv.partition("=")
        t = schema.get(k).type if k in schema else str
        if t is bool: out[k] = v.lower() in ("1", "true", "yes", "y")
        elif t is int: out[k] = int(v)
        elif t is float: out[k] = float(v)
        else: out[k] = v
    return out


def _disp(args, confirm=None, dry=False, auto_confirm=False):
    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(cfg.audit_path) or ".", exist_ok=True)
    reg_probe = build_registry(cfg, Runner())
    confirmer = (lambda a, d: True) if auto_confirm else (confirm or CLIConfirmer(spec_lookup=reg_probe.get))
    d = build_dispatcher(cfg, confirmer, dry_run=dry,
                         on_alert=lambda m: print("ALERT:", m, file=sys.stderr))
    return cfg, d


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="jarvis_actions")
    p.add_argument("-c", "--config", default=os.environ.get("JARVIS_CONFIG"))
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("tools"); s.add_argument("--json", action="store_true")
    for name in ("explain", "run"):
        s = sub.add_parser(name); s.add_argument("action")
        s.add_argument("-a", "--arg", action="append", default=[], metavar="K=V")
        s.add_argument("-d", "--device", default="nitro")
        s.add_argument("--source", choices=[x.name.lower() for x in Source], default="owner")
        s.add_argument("--taint", action="append", default=[], help="mark arg key as tainted")
        if name == "run":
            s.add_argument("--dry-run", action="store_true")
            s.add_argument("-y", "--yes", action="store_true", help="auto-confirm actions")
    s = sub.add_parser("act", help="Natural language utterance executed through Agent")
    s.add_argument("utterance", help="Natural language command")
    s.add_argument("-d", "--device", default="9r")
    s.add_argument("-y", "--yes", action="store_true", help="auto-confirm actions")
    s.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama model")
    s = sub.add_parser("plan"); s.add_argument("file", help="JSON plan file or - for stdin")
    s.add_argument("-d", "--device", default="nitro"); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--source", choices=[x.name.lower() for x in Source], default="owner")
    s = sub.add_parser("audit"); s.add_argument("op", choices=["verify", "tail"]); s.add_argument("path")
    s.add_argument("-n", type=int, default=20)
    s = sub.add_parser("replay"); s.add_argument("path"); s.add_argument("--fail-on-loosened", action="store_true")
    s = sub.add_parser("frida-manifest"); s.add_argument("dir")
    a = p.parse_args(argv)

    if a.cmd == "audit":
        recs = Audit.load(a.path)
        if a.op == "verify":
            ok = Audit.verify(recs); print(f"{len(recs)} records, chain {'OK' if ok else 'BROKEN'}"); return 0 if ok else 2
        for r in recs[-a.n:]:
            print(json.dumps({k: r[k] for k in r if k not in ("prev",)}, default=str))
        return 0
    if a.cmd == "frida-manifest":
        m = build_manifest(a.dir)
        with open(os.path.join(a.dir, "manifest.json"), "w") as f: json.dump(m, f, indent=2)
        print(f"wrote manifest for {len(m)} script(s) -- review scripts before trusting them"); return 0

    cfg = load_config(a.config)
    if a.cmd == "tools":
        reg = build_registry(cfg, Runner())
        rows = [{"name": n, "tier": int(sp.tier), "devices": sp.devices, "summary": sp.summary,
                 "args": {k: v.describe() for k, v in sp.schema.items()}} for n, sp in reg.items()]
        if a.json: print(json.dumps(rows, indent=2))
        else:
            for r in rows: print(f"T{r['tier']}  {r['name']:<18} {r['summary']}")
        return 0
    if a.cmd == "replay":
        reg = build_registry(cfg, Runner())
        rep = replay(Audit.load(a.path), reg, root_devices=cfg.root_devices)
        print(rep.summary())
        for d in rep.diffs: print(f"  seq {d.seq} {d.action}: {d.recorded} -> {d.now} ({d.kind}) {d.reasons}")
        return 3 if (a.fail_on_loosened and rep.loosened) or not rep.chain_ok else 0

    os.makedirs(os.path.dirname(cfg.audit_path) or ".", exist_ok=True)
    if a.cmd == "plan":
        _, d = _disp(a, dry=a.dry_run)
        txt = sys.stdin.read() if a.file == "-" else open(a.file).read()
        try: plan = parse_plan(txt)
        except PlanError as e: print("invalid plan:", e, file=sys.stderr); return 1
        for r in run_plan(d, plan, Source[a.source.upper()], a.device): print(json.dumps(r, default=str))
        return 0
    if a.cmd == "act":
        from .agent import Agent
        cfg, d = _disp(a, dry=False, auto_confirm=a.yes)

        def _call_ollama(system: str, user: str) -> str:
            import urllib.request
            body = json.dumps({
                "model": a.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "format": "json",
                "temperature": 0.0
            }).encode()
            req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                content = data["message"]["content"]
                print(f"[LLM Raw Plan]: {content}", file=sys.stderr)
                return content

        agent = Agent(d, _call_ollama, speak=lambda txt: print(f"JARVIS: {txt}"),
                      devices=list(cfg.devices.keys()), default_device=a.device)
        print(f"User: {a.utterance}")
        res = agent.handle(a.utterance)
        print("\n--- Agent Execution Summary ---")
        print(f"Spoken reply: {res.say}")
        print(f"Turns: {res.turns}")
        print(f"Results: {json.dumps(res.results, indent=2, default=str)}")
        if res.errors:
            print(f"Errors: {res.errors}", file=sys.stderr)
        return 0 if not res.errors else 1

    _, d = _disp(a, dry=getattr(a, "dry_run", False), auto_confirm=getattr(a, "yes", False))
    spec = d.reg.get(a.action)
    act = Action(a.action, _coerce(spec.schema if spec else {}, a.arg), Source[a.source.upper()], a.device,
                 tuple(a.taint))
    if a.cmd == "explain":
        dec = d.explain(act); print(json.dumps(dec.as_dict(), indent=2)); return 0
    print(json.dumps(d.run(act), indent=2, default=str)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
