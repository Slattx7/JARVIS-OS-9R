from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import urllib.request

from . import __version__
from .config import Config, expand, load_config


def _dump_toml(d: dict, prefix: str = "") -> str:
    scalars, tables = [], []
    for k, v in d.items():
        if isinstance(v, dict) and v:
            tables.append((k, v))
        elif v is None:
            scalars.append(f"# {k} = (unset)")
        elif isinstance(v, bool):
            scalars.append(f"{k} = {str(v).lower()}")
        elif isinstance(v, (int, float)):
            scalars.append(f"{k} = {v}")
        elif isinstance(v, dict):
            scalars.append(f"{k} = {{}}")
        else:
            scalars.append(f"{k} = {json.dumps(v)}")
    out = ""
    if prefix and scalars:
        out += f"[{prefix}]\n"
    out += "\n".join(scalars) + ("\n\n" if scalars else "")
    for k, v in tables:
        out += _dump_toml(v, f"{prefix}.{k}" if prefix else k)
    return out


# ---------------- commands ----------------
def cmd_run(args) -> int:
    from .build import build_real

    cfg = load_config(args.config)
    pipe, mic, speaker, oob, conv, stt, tts, llm = build_real(cfg)
    print("warming up models ...")
    stt.warmup()
    tts.warmup()

    async def main():
        await llm.warmup(conv.system)
        speaker.start()
        if oob:
            oob.start()
            print(f"OOB approvals on http://{cfg.oob.bind}:{cfg.oob.port}")
        pipe.on_event = lambda n, d: print(f"[{n}] " + " ".join(f"{k}={v!r}" for k, v in d.items()), flush=True)
        print(f"ready. say '{cfg.wake.model.replace('_', ' ')}'. dry_run={cfg.broker.dry_run}. Ctrl+C to quit.")
        try:
            await pipe.run(mic.frames())
        finally:
            speaker.close()
            if oob:
                oob.stop()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    return 0


def cmd_demo(args) -> int:
    from .audit import verify
    from .testing import GatedMic, TEST_SIGNER, battery_transport, make_stack, silence, utterance
    from .trace import format_report, summarize

    llm = [
        'One moment.\n@@ {"action":"device.battery","args":{"device":"pro14"}}\n',
        "Your phone is at eighty three percent and discharging.",
        'Stopping Spotify.\n@@ {"action":"device.app.force_stop","args":{"device":"pro14","package":"com.spotify.music"}}\n',
        'Running it.\n@@ {"action":"root.shell","args":{"command":"id"}}\n',
    ]
    stt = ["what's the battery on my phone", "force stop spotify on my phone", "yes do it", "run id as root on the nine R"]
    holder: dict = {}
    transport = battery_transport()
    audit_path = args.audit
    if audit_path and os.path.exists(audit_path):
        os.remove(audit_path)
    pipe, _, h = make_stack([], stt, llm, dry_run=args.dry_run, transport=transport, audit_path=audit_path)
    holder["p"] = pipe
    pre = silence(8)
    mic = GatedMic([pre + utterance(), utterance(), utterance(), utterance()], lambda: holder["p"])

    def show(name, d):
        extra = " ".join(f"{k}={v!r}" for k, v in d.items())
        print(f"  [{name}] {extra}")
    pipe.on_event = show

    print(f"jarvis {__version__} demo: fake mic/STT/LLM/TTS/device, real pipeline + broker (dry_run={args.dry_run})\n")
    asyncio.run(pipe.run(mic.frames()))

    print("\nspoken (TTS input):")
    for s in h["tts"].synthesized:
        print(f"  \"{s}\"")
    print("\ndevice commands issued:")
    for c in transport.calls:
        print("  ", " ".join(c))
    pend = h["broker"].list_pending()
    if pend:
        p = pend[0]
        tok = TEST_SIGNER.sign_approval(p["id"], p["nonce"], p["digest"])
        print(f"\nT3 request {p['id']} awaiting out-of-band approval: {p['readback']!r}")
        forged = TEST_SIGNER.sign_approval(p["id"], p["nonce"], "0" * 64)
        print(f"  signature over a different command  -> {h['broker'].approve_oob(p['id'], forged).status}")
        out = h["broker"].approve_oob(p["id"], tok)
        print(f"  phone-side Ed25519 signature        -> {out.status}; device saw: {transport.calls[-1]}")
    print("\nlatency (fake backends, so these measure pipeline overhead only):")
    print(format_report(summarize(pipe.tracer.records)))
    if audit_path:
        ok, n, err = verify(audit_path)
        print(f"\naudit chain: {'OK' if ok else 'BROKEN: ' + str(err)} ({n} entries) -> {audit_path}")
    return 0


def cmd_doctor(args) -> int:
    import importlib

    cfg = load_config(args.config)
    bad = 0

    def line(ok, name, note=""):
        nonlocal bad
        bad += 0 if ok else 1
        print(f"{'ok  ' if ok else 'MISS'} {name:<22}{note}")

    line(sys.version_info >= (3, 11), "python>=3.11", sys.version.split()[0])
    hints = {"numpy": "", "sounddevice": "pip install 'jarvis-local[audio]' (needs portaudio)",
             "faster_whisper": "pip install 'jarvis-local[stt]'", "onnxruntime": "pip install 'jarvis-local[vad]'",
             "openwakeword": "pip install --no-deps openwakeword (its tflite-runtime pin has no wheels on py3.12+)",
             "kokoro": "pip install 'jarvis-local[tts]' (espeak is bundled via espeakng-loader; install system espeak-ng only if it complains)", "cryptography": "optional: faster Ed25519 (pure-Python fallback exists)"}
    for mod, hint in hints.items():
        try:
            m = importlib.import_module(mod)
            line(True, mod, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            line(mod == "cryptography", mod, hint or str(e))
    try:
        from .audio.silero_onnx import find_model

        line(True, "silero model", str(find_model(cfg.vad.model_path)))
    except FileNotFoundError:
        line(False, "silero model", "run `jarvis fetch-models`")
    if cfg.broker.approval_scheme == "ed25519":
        pub = expand(cfg.broker.approver_pub_path)
        line(pub.exists(), "approver.pub", str(pub) if pub.exists() else "T3 (root) actions DISABLED: run `jarvis keygen` on your phone, copy the .pub here")
    for tool, hint in [(cfg.adb_path, "android-tools / platform-tools"), (cfg.frida_ps_path, "pip install frida-tools"),
                       ("espeak-ng", "optional fallback: pacman -S espeak-ng"), ("llama-server", "./scripts/get_llama_cpp.sh (prebuilt) or your own CUDA build")]:
        line(shutil.which(tool) is not None or tool == "espeak-ng", tool, hint if not shutil.which(tool) else shutil.which(tool))
    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True)
        line(r.returncode == 0, "gpu", r.stdout.strip())
    else:
        line(False, "gpu", "nvidia-smi not found")
    try:
        with urllib.request.urlopen(cfg.llm.base_url + "/health", timeout=2) as r:
            line(r.status == 200, "llama-server", f"{cfg.llm.base_url} -> {r.status}")
    except Exception as e:  # noqa: BLE001
        line(False, "llama-server", f"{cfg.llm.base_url} unreachable ({type(e).__name__}); scripts/launch_llama.sh")
    try:
        import sounddevice as sd
        d = sd.query_devices(kind="input")
        line(True, "mic", d["name"])
    except Exception as e:  # noqa: BLE001
        line(False, "mic", str(e)[:60])
    for name in ("pro14", "r9"):
        dev = getattr(cfg, name)
        if shutil.which(cfg.adb_path):
            r = subprocess.run([cfg.adb_path] + (["-s", dev.serial] if dev.serial else []) + ["get-state"],
                               capture_output=True, text=True)
            line(r.stdout.strip() == "device", f"adb:{name}", r.stdout.strip() or r.stderr.strip()[:50])
    return 0 if bad == 0 else 1


def cmd_fetch_models(args) -> int:
    from .audio.silero_onnx import CACHE_PATH, SILERO_SHA256, SILERO_VERSION, fetch_model, sha256_file

    cfg = load_config(args.config)
    if CACHE_PATH.exists() and sha256_file(CACHE_PATH) == SILERO_SHA256 and not args.force:
        print(f"silero vad {SILERO_VERSION}: already present, hash ok ({CACHE_PATH})")
    else:
        p = fetch_model()
        print(f"silero vad {SILERO_VERSION}: downloaded, sha256 verified ({p})")
    try:
        import openwakeword

        openwakeword.utils.download_models([cfg.wake.model])
        print(f"openwakeword '{cfg.wake.model}' + feature models: ready")
    except ImportError:
        print("openwakeword not installed: pip install --no-deps openwakeword (see README: its tflite-runtime pin breaks pip on modern Python)")
    return 0


def _real_backends(cfg, need=("vad", "wake", "tts", "stt", "llm")):
    """Instantiate real backends one by one; a failure is reported, not fatal (so selftest shows everything)."""
    from .audio.vad import make_vad
    from .audio.wake import make_wake
    from .llm import make_llm
    from .stt import make_stt
    from .tts import make_tts

    makers = {"vad": make_vad, "wake": make_wake, "tts": make_tts, "stt": make_stt, "llm": make_llm}
    out, errors = {}, {}
    for name in need:
        try:
            out[name] = makers[name](cfg)
        except Exception as e:  # noqa: BLE001
            errors[name] = f"{type(e).__name__}: {e}"
    return out, errors


def cmd_selftest(args) -> int:
    from .actions import build_registry
    from .bench import run_selftest
    from .llm import build_system_prompt

    cfg = load_config(args.config)
    be, errors = _real_backends(cfg)
    print(f"{'check':<8}{'status':<7}detail")
    shown = set()

    def show(c):
        shown.add(c.name)
        print(f"{c.name:<8}{c.status:<7}{c.detail}", flush=True)

    checks = run_selftest(vad=be.get("vad"), wake=be.get("wake"), tts=be.get("tts"), stt=be.get("stt"), llm=be.get("llm"),
                          system_prompt=build_system_prompt(build_registry(cfg).catalog()), log=show)
    for name, err in errors.items():
        print(f"{name:<8}FAIL   could not load: {err}")
    failed = [c for c in checks if c.status == "FAIL"] + [None] * len(errors)
    print("\n" + ("all checks passed" if not failed else f"{len(failed)} problem(s): fix the first FAIL line, rerun"))
    return 0 if not failed else 1


def cmd_bench(args) -> int:
    from .audio.dsp import read_wav, resample
    from .bench import SAMPLE_TEXT, bench_llm, bench_stt, bench_tts, bench_vad

    cfg = load_config(args.config)
    want = ["vad", "stt", "tts", "llm"] if args.what == "all" else [args.what]
    be, errors = _real_backends(cfg, tuple(w for w in want if w in ("vad", "stt", "tts", "llm")))
    for name, err in errors.items():
        print(f"{name}: could not load: {err}")
    if "vad" in be:
        print("vad:", bench_vad(be["vad"]))
    if "tts" in be:
        print("tts:", bench_tts(be["tts"]))
    if "stt" in be:
        if args.wav:
            pcm = read_wav(args.wav, 16000)
        elif "tts" in be:
            import numpy as np
            pcm = resample(np.concatenate(list(be["tts"].synth(SAMPLE_TEXT))), be["tts"].sample_rate, 16000)
        else:
            print("stt: pass --wav FILE (or include tts so audio can be synthesized)")
            pcm = None
        if pcm is not None:
            print("stt:", bench_stt(be["stt"].transcribe, pcm))
    if "llm" in be:
        from .actions import build_registry
        from .llm import build_system_prompt
        print("llm:", bench_llm(be["llm"], build_system_prompt(build_registry(cfg).catalog())))
    return 1 if errors else 0


def cmd_mic_check(args) -> int:
    import time as _t

    from .audio.dsp import rms
    from .audio.io import SoundDeviceMic
    from .audio.vad import make_vad
    from .calibrate import meter_line

    cfg = load_config(args.config)
    vad = make_vad(cfg)
    mic = SoundDeviceMic(cfg.audio.sample_rate, cfg.audio.frame_samples, cfg.audio.input_device)
    print(f"listening for {args.seconds}s: speak, stay quiet, play the speaker; Ctrl+C to stop\n")

    async def main():
        end, n = _t.time() + args.seconds, 0
        async for f in mic.frames():
            n += 1
            if n % 4 == 0:
                print(meter_line(rms(f), vad.prob(f)), flush=True)
            else:
                vad.prob(f)
            if _t.time() > end:
                break
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    return 0


def cmd_calibrate(args) -> int:
    from .audio.dsp import resample
    from .calibrate import frame_levels, suggest, test_signal, toml_snippet

    try:
        import sounddevice as sd
    except ImportError:
        print("pip install 'jarvis-local[audio]'")
        return 1
    cfg = load_config(args.config)
    sr = cfg.audio.sample_rate

    def rec(seconds, play=None):
        """Record `seconds` from the mic (simultaneously playing `play`, if given); returns per-frame RMS levels."""
        if play is not None:
            buf = sd.playrec(play, samplerate=cfg.tts.sample_rate, channels=1, dtype="int16",
                             device=(cfg.audio.input_device, cfg.audio.output_device))
            sd.wait()
            return frame_levels(resample(buf[:, 0], cfg.tts.sample_rate, sr), cfg.audio.frame_samples)
        buf = sd.rec(int(sr * seconds), samplerate=sr, channels=1, dtype="int16", device=cfg.audio.input_device)
        sd.wait()
        return frame_levels(buf[:, 0], cfg.audio.frame_samples)

    input("1/3  Stay QUIET for 3 seconds (ambient). Press Enter to start... ")
    ambient = rec(3)
    input("2/3  The speaker will play a test voice at your NORMAL listening volume (stay quiet). Press Enter... ")
    sig = test_signal(cfg.tts.sample_rate, 4.0)
    bleed = rec(4, play=sig.reshape(-1, 1))
    input("3/3  Speak normally for 4 seconds (e.g. count slowly). Press Enter, then talk... ")
    speech = [x for x in rec(4) if x > 0.5 * max(ambient + [0.004])]
    s = suggest(ambient, bleed, speech)
    print(f"\nambient p95={s['ambient_p95']}  bleed p95={s['bleed_p95']}  your speech p10={s['speech_p10']}")
    for w in s["warnings"]:
        print("WARNING:", w)
    print("\nsuggested config:\n" + toml_snippet(s))
    return 0


def cmd_wav(args) -> int:
    from .audio.dsp import write_wav
    from .audio.io import WavMic, WavSpeaker
    from .build import build_real
    from .trace import format_report, summarize

    cfg = load_config(args.config)
    if args.no_wake:
        cfg.wake.mode = "always"
    cfg.broker.dry_run = not args.live
    cfg.trace.path = ""
    spk = WavSpeaker(cfg.tts.sample_rate)
    mic = WavMic(args.input, cfg.audio.sample_rate, cfg.audio.frame_samples, tail_s=args.tail)
    pipe, _, _speaker, _oob, conv, _stt, tts, llm = build_real(cfg, mic=mic, speaker=spk)
    spk.sr = tts.sample_rate
    pipe.on_event = lambda n, d: print(f"  [{n}] " + " ".join(f"{k}={v!r}" for k, v in d.items()))

    async def main():
        await llm.warmup(conv.system)
        await pipe.run(mic.frames())
    print(f"{args.input}: {mic.seconds:.1f}s of audio (wake word {'ignored' if args.no_wake else 'required'}; dry_run={cfg.broker.dry_run})")
    asyncio.run(main())
    if args.out:
        write_wav(args.out, spk.audio(), tts.sample_rate)
        print(f"reply audio -> {args.out} ({len(spk.audio()) / tts.sample_rate:.1f}s)")
    print(format_report(summarize(pipe.tracer.records)))
    return 0


def cmd_budget(args) -> int:
    from .budget import AUX_MB, format_plan, llm_mb, plan

    extras = tuple(args.extras or ())
    rows = plan(args.vram_mb, args.reserve_mb, extras)
    print(f"VRAM {args.vram_mb} MB, display reserve {args.reserve_mb} MB, extras {list(extras) or 'none'}\n")
    print(format_plan(rows, only_fits=not args.all, limit=args.limit))
    if args.model:
        b = llm_mb(args.model, args.quant, args.ctx, args.kv)
        print(f"\n{args.model} {args.quant} ctx={args.ctx} kv={args.kv}: " + ", ".join(f"{k}={v:.0f}MB" for k, v in b.items()))
    print(f"\nextras available: {sorted(AUX_MB)}  (estimates +/-15%; verify with llama.cpp load log)")
    return 0


def cmd_trace_report(args) -> int:
    from .trace import format_report, load_records, summarize

    print(format_report(summarize(load_records(args.path or Config().trace.path))))
    return 0


def cmd_audit_verify(args) -> int:
    from .audit import verify

    ok, n, err = verify(args.path or Config().broker.audit_path)
    print(f"{'OK' if ok else 'BROKEN'}: {n} entries" + (f" ({err})" if err else ""))
    return 0 if ok else 2


def cmd_init(args) -> int:
    cfg = Config()
    path = expand(args.config or "~/.config/jarvis/jarvis.toml")
    if path.exists() and not args.force:
        print(f"{path} exists (use --force to overwrite)")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# jarvis config - generated defaults. Unknown keys are errors.\n\n" + _dump_toml(cfg.to_dict()))
        print(f"wrote {path}")
    pub = expand(cfg.broker.approver_pub_path)
    print(f"\nT3 (root) approvals are DISABLED until an approver key exists ({pub} missing)." if not pub.exists()
          else f"\napprover public key: {pub}")
    print("  on your PHONE:  jarvis keygen        # private key never leaves it")
    print(f"  then copy the .pub to this host as {pub}")
    return 0


def cmd_keygen(args) -> int:
    from . import approval, crypto

    if args.hmac:
        approval.generate_hmac_secret(args.private or "~/.config/jarvis/approval.key", args.force)
        print("wrote HMAC secret (legacy scheme; set broker.approval_scheme = \"hmac\"). Share it with the approver device.")
        return 0
    priv = args.private or "~/.config/jarvis/approver.key"
    pub = args.public or "~/.config/jarvis/approver.pub"
    try:
        seed = approval.generate_keypair(priv, pub, args.force)
    except FileExistsError:
        print(f"{expand(priv)} exists; refusing to overwrite (use --force)")
        return 1
    print(f"signing backend: {crypto.backend()}")
    print(f"private key: {expand(priv)} (mode 600): KEEP ON THIS DEVICE, never copy it to the host")
    print(f"public key:  {expand(pub)}  ({crypto.public_key(seed).hex()})")
    print("copy ONLY the .pub file to the host's broker.approver_pub_path (default ~/.config/jarvis/approver.pub)")
    return 0


def cmd_token(args) -> int:
    from .approval import load_signer

    print(load_signer(args.key).sign_approval(args.id, args.nonce, args.digest))
    return 0


def cmd_approve(args) -> int:
    import time

    from .approval import load_signer, make_list_auth

    signer = load_signer(args.key)
    base = args.url.rstrip("/")
    auth = make_list_auth(signer, time.time())
    with urllib.request.urlopen(f"{base}/pending?ts={auth.ts}&sig={auth.sig}", timeout=5) as r:
        pending = json.load(r)["pending"]
    if not pending:
        print("nothing pending")
        return 0
    for p in pending:
        print(f"\nrequest {p['id']} (expires in {p['expires_in']}s)\n  action: {p['action']}\n  args:   {json.dumps(p['args'])}\n  digest: {p['digest'][:16]}...")
        if input("approve this exact action? [y/N] ").strip().lower() != "y":
            continue
        body = json.dumps({"id": p["id"], "token": signer.sign_approval(p["id"], p["nonce"], p["digest"])}).encode()
        req = urllib.request.Request(f"{base}/approve", data=body, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                print("->", json.load(r))
        except urllib.error.HTTPError as e:
            print("->", e.code, e.read().decode())
    return 0


def cmd_prompt(args) -> int:
    from .actions import build_registry
    from .llm import build_system_prompt

    cfg = load_config(args.config)
    print(build_system_prompt(build_registry(cfg).catalog(), cfg.llm.system_prompt))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jarvis", description="Local root-capable personal agent")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("-c", "--config", default=None, help="TOML config path")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="live voice loop").set_defaults(fn=cmd_run)
    d = sub.add_parser("demo", help="end-to-end demo on fake backends (no models/devices needed)")
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--audit", default="/tmp/jarvis_demo_audit.jsonl")
    d.set_defaults(fn=cmd_demo)
    sub.add_parser("doctor", help="check deps, GPU, llama-server, adb").set_defaults(fn=cmd_doctor)
    sub.add_parser("selftest", help="first-run check: TTS->STT round trip, VAD, wake, LLM, with timings").set_defaults(fn=cmd_selftest)
    bn = sub.add_parser("bench", help="benchmark one backend (or all)")
    bn.add_argument("what", choices=["vad", "stt", "tts", "llm", "all"], nargs="?", default="all")
    bn.add_argument("--wav", help="speech file for the stt benchmark")
    bn.set_defaults(fn=cmd_bench)
    mc = sub.add_parser("mic-check", help="live input meter + VAD probability (tune thresholds)")
    mc.add_argument("--seconds", type=int, default=30)
    mc.set_defaults(fn=cmd_mic_check)
    sub.add_parser("calibrate", help="measure ambient/speaker-bleed/speech levels, suggest barge-in config").set_defaults(fn=cmd_calibrate)
    wv = sub.add_parser("wav", help="run the real stack on a wav file (no mic/speaker needed)")
    wv.add_argument("input")
    wv.add_argument("--out", help="write the spoken reply to this wav")
    wv.add_argument("--no-wake", action="store_true", help="skip the wake word (file is just the command)")
    wv.add_argument("--live", action="store_true", help="allow mutating actions (default: dry run)")
    wv.add_argument("--tail", type=float, default=3.0, help="seconds of silence appended")
    wv.set_defaults(fn=cmd_wav)
    fm = sub.add_parser("fetch-models", help="download + hash-verify Silero VAD (pinned) and the wake-word models")
    fm.add_argument("--force", action="store_true")
    fm.set_defaults(fn=cmd_fetch_models)
    b = sub.add_parser("budget", help="VRAM planner for 6 GB")
    b.add_argument("--vram-mb", type=int, default=6144)
    b.add_argument("--reserve-mb", type=int, default=0, help="VRAM held by display/compositor")
    b.add_argument("--extras", nargs="*", default=None, help="e.g. 'stt-gpu (distil-small int8)' kokoro-gpu")
    b.add_argument("--all", action="store_true")
    b.add_argument("--limit", type=int, default=12)
    b.add_argument("--model", default=None)
    b.add_argument("--quant", default="q4_k_m")
    b.add_argument("--ctx", type=int, default=4096)
    b.add_argument("--kv", default="q8_0")
    b.set_defaults(fn=cmd_budget)
    t = sub.add_parser("trace-report", help="latency percentiles from trace.jsonl")
    t.add_argument("path", nargs="?")
    t.set_defaults(fn=cmd_trace_report)
    a = sub.add_parser("audit-verify", help="verify audit hash chain")
    a.add_argument("path", nargs="?")
    a.set_defaults(fn=cmd_audit_verify)
    i = sub.add_parser("init", help="write default config")
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_init)
    g = sub.add_parser("keygen", help="(run on the phone) create the approver keypair")
    g.add_argument("--private")
    g.add_argument("--public")
    g.add_argument("--force", action="store_true")
    g.add_argument("--hmac", action="store_true", help="legacy shared-secret scheme")
    g.set_defaults(fn=cmd_keygen)
    k = sub.add_parser("token", help="compute an approval token offline")
    for n in ("digest", "nonce", "id"):
        k.add_argument(f"--{n}", required=True)
    k.add_argument("--key", required=True)
    k.set_defaults(fn=cmd_token)
    ap_ = sub.add_parser("approve", help="(run on the phone) review + approve pending T3 requests")
    ap_.add_argument("--url", required=True)
    ap_.add_argument("--key", required=True)
    ap_.set_defaults(fn=cmd_approve)
    sub.add_parser("prompt", help="print the generated system prompt").set_defaults(fn=cmd_prompt)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
