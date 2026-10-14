#!/usr/bin/env python3
"""activate_cmd example -- UNTESTED in the build sandbox (no ollama there). Verify against your Ollama version.

Makes `jarvis:<version>` serve <base> (+ this version's LoRA adapter when the artifact is an adapter dir).
loop.json:  "activate_cmd": ["python", "examples/activate_ollama.py", "{version}", "{artifact}", "<ollama base tag>"]
Replay then goes through the SAME stack that serves Jarvis (quantised GGUF base + adapter), not the training-time
NF4 model -- which is exactly what the gate should measure.
"""
import pathlib
import subprocess
import sys
import tempfile

version, artifact, base = sys.argv[1:4]
modelfile = f"FROM {base}\n"
if pathlib.Path(artifact).is_dir():              # v0 bootstrap artifact is a base-model id, not a dir
    modelfile += f"ADAPTER {artifact}\n"
with tempfile.TemporaryDirectory() as d:
    mf = pathlib.Path(d) / "Modelfile"
    mf.write_text(modelfile)
    subprocess.run(["ollama", "create", f"jarvis:{version}", "-f", str(mf)], check=True)
