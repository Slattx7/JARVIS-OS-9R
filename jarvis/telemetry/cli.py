"""Command-line entrypoint.

Subcommands: collect, watch, train, score, synth, permissions, trackers, probes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from .collector import Collector, CollectionError, DeviceNotFoundError
from .explain import explain_in_plain_english, explain_session
from .features import FRIDA_DEFAULT_PORT, LOOPBACK_HOSTS, extract_session_features
from .labels import LabelError, build_training_set
from .model import TrainingError, load_model, save_model, score_session, train_classifier
from .permissions import assess_permissions, load_declared_permissions
from .schema import EventType
from .storage import TraceStore
from .streaming import StreamingScorer
from .trackers import find_known_trackers, hosts_by_package, shared_domains, similar_apps


# --------------------------------------------------------------------------
# parser construction
# --------------------------------------------------------------------------

def _add_collect_parser(subparsers) -> None:
    p = subparsers.add_parser("collect", help="Collect a behavioral trace from one app on the 9R.")
    p.add_argument("package_name", help="Target app package, e.g. com.example.app")
    p.add_argument("--out", default="traces.jsonl", help="Path to the JSONL trace log")
    p.add_argument("--duration", type=float, default=None, help="Seconds to collect (default: until Ctrl-C)")
    p.add_argument("--mode", choices=["spawn", "attach"], default="spawn")
    p.add_argument("-v", "--verbose", action="store_true")


def _add_watch_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "watch",
        help="Collect from one app while scoring it live, alerting when its risk drifts from its own baseline.",
    )
    p.add_argument("package_name")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="traces.jsonl", help="Raw events are still written here as they arrive")
    p.add_argument("--duration", type=float, default=None, help="Seconds to watch (default: until Ctrl-C)")
    p.add_argument("--mode", choices=["spawn", "attach"], default="spawn")
    p.add_argument("--window-s", type=float, default=60.0, help="Rolling window length in seconds")
    p.add_argument("--cumulative", action="store_true", help="Score the whole session so far instead of a rolling window")
    p.add_argument("--baseline-events", type=int, default=10)
    p.add_argument("--drift-threshold", type=float, default=0.3)
    p.add_argument("--score-every", type=int, default=5, help="Re-score every N events (CPU vs. latency)")
    p.add_argument("-v", "--verbose", action="store_true")


def _add_train_parser(subparsers) -> None:
    p = subparsers.add_parser("train", help="Train the GBT baseline on labeled trace logs.")
    p.add_argument("--logs", nargs="+", required=True, help="One or more JSONL trace log files")
    p.add_argument("--labels", required=True, help="JSON file: {package_name: 0|1}, 1=malicious")
    p.add_argument("--out", default="model.joblib")


def _add_score_parser(subparsers) -> None:
    p = subparsers.add_parser("score", help="Score a session's trace log with a trained model.")
    p.add_argument("session_log", help="JSONL trace log containing exactly one session")
    p.add_argument("--model", required=True)
    p.add_argument("--explain", action="store_true", help="Print the top SHAP-contributing features")
    p.add_argument("--narrate", action="store_true", help="Explain via a local Ollama model instead of raw SHAP")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--ollama-model", default="qwen2.5")


def _add_synth_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "synth",
        help="Generate a synthetic labeled dataset for testing the pipeline (NOT real data).",
    )
    p.add_argument("--out-dir", default="synthetic_traces")
    p.add_argument("--labels-out", default="synthetic_labels.json")
    p.add_argument("--n-benign", type=int, default=20)
    p.add_argument("--n-malicious", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)


def _add_permissions_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "permissions",
        help="Compare an app's declared dangerous permissions against the AppOps it actually exercised.",
    )
    p.add_argument("session_log", help="JSONL trace log(s) for the app")
    p.add_argument("--declared", required=True, help="File containing the app's android.permission.* names (dumpsys/aapt output, JSON, or one per line)")
    p.add_argument("--package", default=None, help="Package to analyse (required if the log holds several)")


def _add_trackers_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "trackers",
        help="Known-tracker hits, widely shared endpoints, and apps that likely share an embedded SDK.",
    )
    p.add_argument("--logs", nargs="+", required=True)
    p.add_argument("--min-apps", type=int, default=2, help="A domain must be contacted by this many apps to count as shared")
    p.add_argument("--min-jaccard", type=float, default=0.3)
    p.add_argument("--top", type=int, default=20, help="Max rows to print per section (real fleets produce a lot of pairs)")


def _add_probes_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "probes",
        help="List the root/Frida/debugger checks an app made - i.e. whether it was checking if it's being watched.",
    )
    p.add_argument("session_log")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="warden",
        description="Frida-based Android app behavioral risk classifier.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_collect_parser(subparsers)
    _add_watch_parser(subparsers)
    _add_train_parser(subparsers)
    _add_score_parser(subparsers)
    _add_synth_parser(subparsers)
    _add_permissions_parser(subparsers)
    _add_trackers_parser(subparsers)
    _add_probes_parser(subparsers)
    return parser


# --------------------------------------------------------------------------
# command handlers
# --------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _run_collect(args) -> int:
    _configure_logging(args.verbose)
    store = TraceStore(args.out)
    collector = Collector(store)
    try:
        session_id = collector.run(package_name=args.package_name, duration_s=args.duration, mode=args.mode)
    except (DeviceNotFoundError, CollectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"session {session_id} complete -> {args.out}")
    return 0


def _run_watch(args) -> int:
    _configure_logging(args.verbose)
    try:
        model = load_model(args.model)
        scorer = StreamingScorer(
            model,
            window_s=None if args.cumulative else args.window_s,
            baseline_events=args.baseline_events,
            drift_threshold=args.drift_threshold,
            score_every=args.score_every,
        )
    except (TrainingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    state = {"baseline_announced": False, "alerts": 0}

    def on_event(event) -> None:
        alert = scorer.ingest(event)
        if not state["baseline_announced"] and scorer.baseline_score is not None:
            print(f"baseline established: risk={scorer.baseline_score:.3f}", flush=True)
            state["baseline_announced"] = True
        if alert is not None:
            state["alerts"] += 1
            print(
                f"DRIFT at {alert.timestamp.isoformat()}: "
                f"risk {alert.baseline_score:.3f} -> {alert.current_score:.3f} (delta {alert.delta:+.3f})",
                flush=True,
            )

    collector = Collector(TraceStore(args.out))
    try:
        session_id = collector.run(
            package_name=args.package_name,
            duration_s=args.duration,
            mode=args.mode,
            on_event=on_event,
        )
    except (DeviceNotFoundError, CollectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    last = scorer.last_score
    last_text = "n/a" if last is None else f"{last:.3f}"
    print(f"session {session_id} complete -> {args.out}; drift alerts={state['alerts']}; last risk={last_text}")
    return 0


def _run_train(args) -> int:
    try:
        rows, y, session_ids = build_training_set(args.logs, args.labels)
        model, metrics = train_classifier(rows, y)
    except (LabelError, TrainingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    save_model(model, args.out)
    print(f"trained on {len(rows)} labeled sessions -> {args.out}")
    print(json.dumps(metrics, indent=2))
    return 0


def _run_score(args) -> int:
    try:
        model = load_model(args.model)
    except TrainingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    events = TraceStore(args.session_log).read_all()
    if not events:
        print(f"error: no events in {args.session_log}", file=sys.stderr)
        return 1

    session_ids = {e.session_id for e in events}
    if len(session_ids) > 1:
        print(
            f"error: {args.session_log} contains {len(session_ids)} sessions; score expects exactly one",
            file=sys.stderr,
        )
        return 1

    features = extract_session_features(events)
    row = features.to_row()
    risk = score_session(model, row)
    print(f"package={features.package_name} risk_score={risk:.3f}")

    if args.explain or args.narrate:
        contributions = explain_session(model, row)
        if args.narrate:
            print(explain_in_plain_english(
                features.package_name, risk, contributions,
                ollama_url=args.ollama_url, ollama_model=args.ollama_model,
            ))
        else:
            for c in contributions:
                sign = "+" if c.shap_value >= 0 else ""
                print(f"  {c.feature}={c.value:g}  ({sign}{c.shap_value:.3f})")

    return 0


def _run_synth(args) -> int:
    from .synthetic import generate_synthetic_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events, labels = generate_synthetic_dataset(
        n_benign=args.n_benign, n_malicious=args.n_malicious, seed=args.seed,
    )

    by_session: dict[str, list] = {}
    for e in events:
        by_session.setdefault(e.session_id, []).append(e)

    for session_events in by_session.values():
        pkg = session_events[0].package_name
        TraceStore(out_dir / f"{pkg}.jsonl").append_many(session_events)

    Path(args.labels_out).write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"wrote {len(by_session)} synthetic sessions -> {out_dir}/, labels -> {args.labels_out}")
    return 0


def _run_permissions(args) -> int:
    events = TraceStore(args.session_log).read_all()
    if not events:
        print(f"error: no events in {args.session_log}", file=sys.stderr)
        return 1

    packages = sorted({e.package_name for e in events})
    if args.package is not None:
        if args.package not in packages:
            print(f"error: package {args.package!r} not in log (found: {', '.join(packages)})", file=sys.stderr)
            return 1
        package = args.package
    elif len(packages) == 1:
        package = packages[0]
    else:
        print(f"error: log holds several packages ({', '.join(packages)}); pick one with --package", file=sys.stderr)
        return 1

    declared_path = Path(args.declared)
    if not declared_path.exists():
        print(f"error: declared-permissions file not found: {declared_path}", file=sys.stderr)
        return 1

    declared = load_declared_permissions(declared_path)
    usage = assess_permissions(package, declared, [e for e in events if e.package_name == package])

    print(f"package={usage.package_name}")
    print(f"declared dangerous permissions: {len(usage.declared_dangerous)}")
    print(f"  exercised:   {', '.join(usage.exercised) or '-'}")
    print(f"  unexercised: {', '.join(usage.unexercised) or '-'}  ({usage.unexercised_ratio:.0%} of declared)")
    print(f"exercised but NOT declared: {', '.join(usage.exercised_but_undeclared) or '-'}")
    print(
        f"app-op events: {usage.total_op_events} total, "
        f"{usage.unmapped_op_events} on non-tracked ops, {usage.numeric_op_events} with numeric identifiers"
    )
    if usage.numeric_op_events:
        print(
            "warning: numeric op identifiers can't be decoded without your build's op-code table, "
            "so the unexercised list may be overstated."
        )
    if usage.unexercised:
        print("note: 'unexercised' only reflects the collected session(s); a short run can miss legitimate later use.")
    return 0


def _run_trackers(args) -> int:
    events = []
    for path in args.logs:
        events.extend(TraceStore(path).iter_events())
    host_map = hosts_by_package(events)
    if not host_map:
        print("no network destinations found in the given logs")
        return 0

    print("== known trackers ==")
    any_known = False
    for package in sorted(host_map):
        hits = find_known_trackers(host_map[package])
        for host, (suffix, category) in sorted(hits.items()):
            any_known = True
            print(f"  {package}: {host} [{category}, matches {suffix}]")
    if not any_known:
        print("  none matched the built-in list")

    print(f"== domains contacted by >= {args.min_apps} apps ==")
    shared = shared_domains(host_map, min_apps=args.min_apps)
    if not shared:
        print("  none")
    for item in shared[: args.top]:
        tag = f" [{item.known_category}]" if item.known_category else " [unlisted - candidate SDK endpoint]"
        print(f"  {item.domain}{tag}: {len(item.packages)} apps ({', '.join(item.packages)})")
    if len(shared) > args.top:
        print(f"  ... {len(shared) - args.top} more (raise --top to see them)")

    print(f"== app pairs with domain overlap >= {args.min_jaccard:.2f} ==")
    pairs = similar_apps(host_map, min_jaccard=args.min_jaccard)
    if not pairs:
        print("  none")
    for pair in pairs[: args.top]:
        print(f"  {pair.package_a} ~ {pair.package_b}: jaccard={pair.jaccard:.2f} via {', '.join(pair.shared_domains)}")
    if len(pairs) > args.top:
        print(f"  ... {len(pairs) - args.top} more (raise --top to see them)")
    return 0


def _run_probes(args) -> int:
    events = TraceStore(args.session_log).read_all()
    if not events:
        print(f"error: no events in {args.session_log}", file=sys.stderr)
        return 1

    counts: Counter = Counter()
    for e in events:
        if e.event_type == EventType.ENV_PROBE:
            counts[(e.details.get("category"), e.details.get("kind"), e.details.get("target"))] += 1
        elif (
            e.event_type == EventType.NETWORK_CONNECT
            and e.details.get("port") == FRIDA_DEFAULT_PORT
            and e.details.get("host") in LOOPBACK_HOSTS
        ):
            counts[("frida", "loopback_connect", f"{e.details.get('host')}:{FRIDA_DEFAULT_PORT}")] += 1

    if not counts:
        print("no environment probes observed")
        return 0

    print(f"{sum(counts.values())} probe events, {len(counts)} distinct:")
    for (category, kind, target), n in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print(f"  x{n}  [{category}] {kind}: {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "collect": _run_collect,
        "watch": _run_watch,
        "train": _run_train,
        "score": _run_score,
        "synth": _run_synth,
        "permissions": _run_permissions,
        "trackers": _run_trackers,
        "probes": _run_probes,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
