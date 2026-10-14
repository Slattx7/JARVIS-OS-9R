"""Static HTML renderers. One file each, inline CSS, no JavaScript, no external requests, every dynamic string is
html-escaped (trace data contains user speech and model output: treat it as hostile).

    render_trace_html(trace)    waterfall of spans with tier/status/frozen badges and collapsible inputs/outputs
    render_report_html(report)  replay verdict, scorer table, failing cases with baseline-vs-candidate actions
"""
from __future__ import annotations

import datetime as dt
import html
import json
from typing import Any, Dict, List, Optional

from .harness import Report
from .schema import Span, Trace

_CSS = """
:root{--bg:#fff;--fg:#1b1f24;--mut:#6a737d;--line:#e1e4e8;--card:#f6f8fa;--ok:#1a7f37;--bad:#cf222e;--warn:#9a6700;
--llm:#8250df;--asr:#0969da;--recall:#1b7c83;--permission:#bf3989;--confirm:#bc4c00;--action:#cf222e;--tts:#2da44e;
--wake:#6a737d;--context:#8c959f;--risk:#d4a72c;--custom:#57606a}
@media(prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;--line:#30363d;--card:#161b22;
--ok:#3fb950;--bad:#f85149;--warn:#d29922}}
*{box-sizing:border-box}body{margin:0;padding:20px;background:var(--bg);color:var(--fg);
font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin-inline:auto}
h1{font-size:20px;margin:0 0 6px}h2{font-size:16px;margin:26px 0 8px}h4{margin:8px 0 4px;font-size:13px}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
pre{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px;overflow:auto;max-height:260px;margin:4px 0}
.meta{color:var(--mut);margin:0 0 14px}.badge{display:inline-block;border-radius:10px;padding:1px 8px;font-size:12px;
border:1px solid var(--line);background:var(--card)}
.ok,.pass{color:var(--ok)}.error,.fail,.safety_fail,.blocked,.orphaned{color:var(--bad)}.running{color:var(--warn)}
table{border-collapse:collapse;width:100%}td,th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left;
vertical-align:top}th{color:var(--mut);font-weight:600;font-size:12px}
.wf td.lbl{white-space:nowrap;width:34%;overflow:hidden;text-overflow:ellipsis;max-width:380px}
.wf td.bar{width:52%;position:relative}.wf .b{height:10px;border-radius:3px;min-width:2px;margin-top:5px}
.wf td.dur{width:14%;text-align:right;color:var(--mut);white-space:nowrap}
.k{display:inline-block;min-width:68px;font-size:11px;color:#fff;border-radius:3px;padding:0 5px;margin-right:6px;text-align:center}
.k-llm{background:var(--llm)}.k-asr{background:var(--asr)}.k-recall{background:var(--recall)}
.k-permission{background:var(--permission)}.k-confirm{background:var(--confirm)}.k-action{background:var(--action)}
.k-tts{background:var(--tts)}.k-wake{background:var(--wake)}.k-context{background:var(--context)}
.k-risk{background:var(--risk)}.k-custom{background:var(--custom)}
.b.k-llm{background:var(--llm)}.b.k-asr{background:var(--asr)}.b.k-recall{background:var(--recall)}
.b.k-permission{background:var(--permission)}.b.k-confirm{background:var(--confirm)}.b.k-action{background:var(--action)}
.b.k-tts{background:var(--tts)}.b.k-wake{background:var(--wake)}.b.k-context{background:var(--context)}
.b.k-risk{background:var(--risk)}.b.k-custom{background:var(--custom)}
.tag{font-size:11px;border:1px solid var(--line);border-radius:3px;padding:0 4px;margin-left:4px;color:var(--mut)}
.t2,.t3{color:var(--bad);border-color:var(--bad)}
.banner{border-radius:8px;padding:10px 14px;font-weight:600;border:1px solid var(--line);background:var(--card)}
.banner.pass{border-color:var(--ok)}.banner.fail{border-color:var(--warn)}.banner.safety_fail{border-color:var(--bad)}
details.case{border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:6px 10px;background:var(--card)}
details.case summary{cursor:pointer}.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.cols ol{margin:2px 0;padding-left:20px}.cols li{font-family:ui-monospace,monospace;font-size:12px}
ul.f{margin:6px 0;padding-left:18px}
"""


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _ts(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1e9).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _ms(ns: Optional[int]) -> str:
    return "-" if ns is None else f"{ns / 1e6:.1f} ms"


def _json(x: Any, n: int = 4000) -> str:
    try:
        s = json.dumps(x, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = repr(x)
    return s if len(s) <= n else s[:n] + f"\n… ({len(s) - n} more characters)"


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
            f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>{body}</body></html>")


def _layout(tr: Trace) -> Dict[str, Any]:
    """span_id -> (x_ns, dur_ns, depth). Live traces use real offsets; replay traces (frozen spans have no real
    timing) are laid out sequentially."""
    ids = {s.span_id for s in tr.spans}
    kids: Dict[Optional[str], List[Span]] = {}
    for s in tr.spans:
        parent = s.parent_id if s.parent_id in ids else None
        kids.setdefault(parent, []).append(s)
    pos: Dict[str, tuple] = {}
    total = 0
    if "replay" in tr.tags:
        def place(s: Span, x: int, d: int) -> int:
            dur = s.dur_ns or 0
            pos[s.span_id] = (x, dur, d)
            cur = x
            for c in kids.get(s.span_id, []):
                cur = place(c, cur, d + 1)
            return max(x + dur, cur)

        cursor = 0
        for r in kids.get(None, []):
            cursor = place(r, cursor, 0)
        total = cursor
    else:
        def walk(s: Span, d: int) -> None:
            pos[s.span_id] = (s.t_start_ns - tr.t_start_ns, s.dur_ns or 0, d)
            for c in kids.get(s.span_id, []):
                walk(c, d + 1)

        for r in kids.get(None, []):
            walk(r, 0)
        total = max([x + dur for x, dur, _ in pos.values()] + [tr.dur_ns or 0])
    return {"pos": pos, "total": max(total, 1), "kids": kids}


def render_trace_html(tr: Trace, title: Optional[str] = None) -> str:
    lay = _layout(tr)
    pos, total = lay["pos"], lay["total"]
    rows: List[str] = []
    for s in tr.spans:
        x, dur, depth = pos.get(s.span_id, (0, 0, 0))
        tags = []
        t = s.meta.get("tier")
        if t is not None:
            tags.append(f"<span class=\"tag t{_e(t)}\">T{_e(t)}</span>")
        for flag, txt in (("frozen", "frozen"), ("stub", "STUB"), ("dry_run", "dry-run"), ("irreversible", "irreversible")):
            if s.meta.get(flag):
                tags.append(f"<span class=\"tag\">{txt}</span>")
        if s.meta.get("undo"):
            tags.append("<span class=\"tag\">undo</span>")
        if s.meta.get("snapshot"):
            tags.append("<span class=\"tag\">snapshot</span>")
        if s.status == "error" and s.error:
            tags.append(f"<span class=\"tag error\">{_e(s.error.get('type'))}</span>")
        left, width = x / total * 100, max(dur / total * 100, 0.4)
        rows.append(
            f"<tr class=\"span\"><td class=\"lbl\" style=\"padding-left:{6 + depth * 16}px\">"
            f"<span class=\"k k-{_e(s.kind)}\">{_e(s.kind)}</span>{_e(s.name)} "
            f"<span class=\"{_e(s.status)}\">{_e(s.status)}</span>{''.join(tags)}</td>"
            f"<td class=\"bar\"><div class=\"b k-{_e(s.kind)}\" style=\"margin-left:{left:.2f}%;width:{min(width, 100 - left):.2f}%\">"
            f"</div></td><td class=\"dur\">{_ms(s.dur_ns)}</td></tr>"
            f"<tr><td colspan=\"3\" style=\"padding-left:{6 + depth * 16}px\"><details><summary>inputs / outputs"
            f"</summary><pre>in:  {_e(_json(s.inputs))}\nout: {_e(_json(s.outputs))}"
            f"{('' if not s.meta else chr(10) + 'meta: ' + _e(_json(s.meta)))}</pre></details></td></tr>")
    labels = ", ".join(sorted(tr.label_set()))
    head = (f"<h1>{_e(tr.name)} <span class=\"badge {_e(tr.status)}\">{_e(tr.status)}</span></h1>"
            f"<p class=\"meta\">{_e(tr.trace_id)} · {_e(_ts(tr.t_start_ns))} · {_e(_ms(tr.dur_ns))} · "
            f"est. pre-TTS {tr.pre_tts_ms():.1f} ms · tags: {_e(', '.join(tr.tags)) or '-'} · labels: {_e(labels) or '-'}"
            f"{' · layout: sequential (replay)' if 'replay' in tr.tags else ''}</p>")
    io = (f"<details><summary>trace inputs / outputs</summary><pre>in:  {_e(_json(tr.inputs))}\n"
          f"out: {_e(_json(tr.outputs))}</pre></details>")
    return _page(title or f"trace {tr.trace_id}", head + io + f"<table class=\"wf\">{''.join(rows)}</table>")


def render_report_html(rep: Report) -> str:
    verdict = rep.verdict
    scorers = "".join(
        f"<tr><td>{_e(r['scorer'])}</td><td>{_e(r['severity'])}</td><td>{r['passed']}/{r['n']}</td>"
        f"<td>{r['mean_score']:.3f}</td></tr>" for r in rep.scorer_table())
    lat = rep.latency()
    cases: List[str] = []
    for c in sorted(rep.cases, key=lambda c: {"safety_fail": 0, "error": 1, "fail": 2, "pass": 3}[c.verdict]):
        failing = [s for s in c.scores if not s.passed and s.severity != "info"]
        fl = "".join(f"<li><b>{_e(s.scorer)}</b> [{_e(s.severity)}]: {_e(s.detail)}</li>" for s in failing)
        if c.error:
            fl += f"<li><b>agent error</b>: {_e(c.error)}</li>"
        div = "".join(f"<li>{_e(d['kind'])}/{_e(d['name'])}: {_e(d['reason'])}</li>" for d in c.divergences)
        base = "".join(f"<li>{_e(a)}</li>" for a in c.baseline_actions) or "<li>(none)</li>"
        cand = "".join(f"<li>{_e(a)}</li>" for a in c.candidate_actions) or "<li>(none)</li>"
        cases.append(
            f"<details class=\"case\"{' open' if c.verdict != 'pass' else ''}><summary>"
            f"<span class=\"badge {_e(c.verdict)}\">{_e(c.verdict)}</span> {_e(c.name)} "
            f"<code>{_e(c.trace_id)}</code>{f' · {len(failing)} failing' if failing else ''}</summary>"
            f"{('<ul class=f>' + fl + '</ul>') if fl else ''}"
            f"<div class=\"cols\"><div><h4>baseline actions</h4><ol>{base}</ol></div>"
            f"<div><h4>candidate actions</h4><ol>{cand}</ol></div></div>"
            f"{('<h4>divergences</h4><ul class=f>' + div + '</ul>') if div else ''}"
            f"<p class=\"meta\">pre-TTS est. {c.baseline_pre_tts_ms:.0f} ms → {c.candidate_pre_tts_ms:.0f} ms"
            f"{' · candidate trace ' + _e(c.candidate_trace_id) if c.candidate_trace_id else ''}</p></details>")
    body = (f"<h1>Replay report: {_e(rep.label)}</h1>"
            f"<p class=\"meta\">live kinds: {_e(', '.join(rep.live))} · on_miss: {_e(rep.on_miss)} · "
            f"{len(rep.skipped)} baseline trace(s) skipped</p>"
            f"<div class=\"banner {_e(verdict)}\">{_e(rep.summary_line())}</div>"
            f"<h2>Scorers</h2><table><tr><th>scorer</th><th>severity</th><th>passed</th><th>mean score</th></tr>{scorers}</table>"
            f"<h2>Latency (estimated wake → first audio)</h2><table><tr><th></th><th>p50</th><th>p95</th></tr>"
            f"<tr><td>baseline</td><td>{lat['baseline_p50']:.0f} ms</td><td>{lat['baseline_p95']:.0f} ms</td></tr>"
            f"<tr><td>candidate</td><td>{lat['candidate_p50']:.0f} ms</td><td>{lat['candidate_p95']:.0f} ms</td></tr></table>"
            f"<h2>Cases ({len(rep.cases)})</h2>{''.join(cases)}")
    return _page(f"replay report: {rep.label}", body)
