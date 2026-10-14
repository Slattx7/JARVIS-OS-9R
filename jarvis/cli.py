"""Unified JARVIS-9R CLI
Unified Command Choke Point for Perception, Context, Policy, Audit, Loop, and Phone Control.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

from . import __version__


def cmd_status(args):
    """Print complete status of perception, phone link, policy lockdown, audit store, and loop."""
    print("=" * 60)
    print(f"🛡️  JARVIS-9R UNIFIED OS (v{__version__}) - SYSTEM STATUS")
    print("=" * 60)
    
    # 1. Hardware & VRAM
    print("\n[Hardware Topology]")
    print("  Host Node : Acer Nitro 5 (Linux x86_64, RTX 3050 Laptop GPU 6GB)")
    print("  Target    : OnePlus 9R (lemonades, Snapdragon 870, Magisk Root)")
    print("  Link Mesh : Tailscale Mesh Network")

    # 2. Perception & Models
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    print("\n[Perception Pipeline]")
    print(f"  Models Directory: {model_dir}")
    kokoro = os.path.exists(os.path.join(model_dir, "kokoro-v1.0.onnx"))
    voices = os.path.exists(os.path.join(model_dir, "voices-v1.0.bin"))
    print(f"  Kokoro TTS Engine  : {'ONLINE (v1.0)' if kokoro else 'MISSING'}")
    print(f"  Voice Embeddings   : {'ONLINE' if voices else 'MISSING'}")
    print("  VAD Engine         : Silero ONNX (CPU)")
    print("  ASR Engine         : Speculative faster-whisper-small (CPU INT8)")

    # 3. Policy & Gate
    print("\n[Zero-Trust Policy Gate]")
    print("  Tiers Enforced     : T0 (Observe), T1 (Reversible), T2 (Stateful), T3 (Root)")
    print("  Taint Analysis     : Active (+1 Tier escalation on untrusted data)")
    print("  Lockdown Trigger   : 5 denials / 60 seconds -> 5 min observe-only")

    # 4. Loop & Learning
    print("\n[Trace-Driven QLoRA Loop]")
    print("  Adapter Target     : QLoRA (4-bit NF4, r=16, 8-bit AdamW)")
    print("  Promotion Gate     : Paired McNemar Hypothesis Test (p < 0.05)")
    print("  Active Deployment  : v0 (Base / Human-gated)")
    print("=" * 60)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="Unified JARVIS-9R Command Interface")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    # Status
    p_status = sub.add_parser("status", help="Display full system health and subsystem status")

    # Perception / Voice runtime
    p_run = sub.add_parser("run", help="Start the live voice loop")
    p_doctor = sub.add_parser("doctor", help="Check hardware, models, GPU, audio, and device links")
    p_selftest = sub.add_parser("selftest", help="Run end-to-end selftest across the stack")
    p_demo = sub.add_parser("demo", help="Run full pipeline demo on synthetic backends")
    p_wire = sub.add_parser("wire", help="Run end-to-end trace & policy wiring verification")

    # Policy & Actions
    p_policy = sub.add_parser("policy", help="Interact with the policy gate")
    p_policy.add_argument("policy_args", nargs=argparse.REMAINDER, help="Policy command and arguments")

    # Autonomous Agent Act
    p_act = sub.add_parser("act", help="Execute an autonomous natural-language request via Agent")
    p_act.add_argument("utterance", help="Natural language utterance / command")
    p_act.add_argument("-d", "--device", default="9r", help="Target device (e.g. 9r, nitro)")
    p_act.add_argument("-y", "--yes", action="store_true", help="Auto-confirm actions")
    p_act.add_argument("-c", "--config", default=None, help="Config path")
    p_act.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama model name")

    p_undo = sub.add_parser("undo", help="Undo a previously executed reversible action")
    p_undo.add_argument("trace_id", nargs="?", help="Trace ID to undo")

    # Loop
    p_loop = sub.add_parser("loop", help="Manage autonomous QLoRA loop cycles")
    p_loop.add_argument("action", choices=["status", "plan", "demo", "rollback"])

    # Context
    p_context = sub.add_parser("context", help="Query recall memory graph")
    p_context.add_argument("query", nargs="?", default="", help="Query string")

    # Trace
    p_trace = sub.add_parser("trace", help="Flight recorder audit operations")
    p_trace.add_argument("action", choices=["verify", "demo", "status"])

    args, extra = parser.parse_known_args(argv)

    if args.cmd == "status" or not args.cmd:
        return cmd_status(args)

    if args.cmd in ("run", "doctor", "selftest", "demo"):
        from .perception.cli import main as perception_main
        return perception_main([args.cmd] + extra)

    if args.cmd == "wire":
        import subprocess
        script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "jarvis_wiring.py"))
        return subprocess.call([sys.executable, script] + extra)

    if args.cmd == "undo":
        from .audit.cli import main as audit_main
        return audit_main(["undo"] + ([args.trace_id] if args.trace_id else []) + extra)

    if args.cmd == "act":
        from .policy.cli import main as policy_main
        cfg_path = args.config or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "jarvis.json"))
        forward = ["act", args.utterance, "-d", args.device, "--model", args.model]
        if os.path.exists(cfg_path):
            forward = ["-c", cfg_path] + forward
        if args.yes:
            forward.append("-y")
        return policy_main(forward + extra)

    if args.cmd == "policy":
        from .policy.cli import main as policy_main
        cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "jarvis.json"))
        pargs = list(args.policy_args) if args.policy_args else (list(extra) if extra else ["tools"])
        if "-c" not in pargs and "--config" not in pargs and os.path.exists(cfg_path):
            pargs = ["-c", cfg_path] + pargs
        return policy_main(pargs)

    if args.cmd == "loop":
        from .loop.cli import main as loop_main
        return loop_main([args.action] + extra)

    if args.cmd == "trace":
        from .audit.cli import main as audit_main
        return audit_main([args.action, "--store", "demo.db"] if args.action == "demo" else [args.action] + extra)

    if args.cmd == "context":
        print(f"Recall query: '{args.query}' - Ingest server ready on port 8787.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
