# 🧠 JARVIS-OS-9R: Autonomous Edge AI Operating System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Target Device](https://img.shields.io/badge/Device-OnePlus_9R_(lemonades)-FF0000.svg?style=for-the-badge&logo=android&logoColor=white)](https://lineageos.org/)
[![Acceleration](https://img.shields.io/badge/Compute-NVIDIA_RTX_3050_GPU-76B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/)
[![Kernel](https://img.shields.io/badge/Ring--0-eBPF_Telemetry-black.svg?style=for-the-badge&logo=linux&logoColor=white)](https://ebpf.io/)
[![Tests](https://img.shields.io/badge/Test_Suite-282%2F282_Passing-success.svg?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

**A unified, zero-trust edge operating system orchestrating mobile device autonomy, local speech perception, in-kernel telemetry, and self-improving policy execution.**

</div>

---

## 🌟 Executive Overview

**JARVIS-OS-9R** represents the convergence of autonomous mobile robotics, local AI perception, and ring-0 kernel security. Designed specifically for the **OnePlus 9R** (`lemonades`, Snapdragon 870) tethered over an encrypted **Tailscale mesh** to an **RTX 3050 workstation host**, JARVIS-OS operates with sub-second voice loops, strict cryptographic audit chains, and in-kernel active mitigation.

Unlike cloud-dependent voice assistants or fragile user-space macro tools, JARVIS-OS enforces:
1. **Zero-Trust Permission Tiers (Tier-0 to Tier-3):** No autonomous action touches device hardware, system files, or sensitive APIs without taint analysis and cryptographic validation.
2. **Deterministic Cryptographic Audit Trail:** Every perception, reasoning step, and device action is appended to a tamper-proof SHA-256 hash-chained ledger.
3. **Hybrid Edge-Workstation Split:** Low-latency sensor ingestion and execution run directly on the phone (<14MB RAM, <0.2% CPU); heavy neural reasoning (Ollama Qwen 2.5 7B, Whisper, Kokoro) executes on the host GPU.
4. **Ring-0 Kernel Grounding:** Telemetry does not rely on easily circumvented Android userland hooks—it hooks directly into kernel syscalls via eBPF probes.

---

## 🏛️ System Architecture

```mermaid
graph TD
    subgraph "Edge Target: OnePlus 9R (lemonades)"
        MIC[Hardware Microphone] --> VAD[Silero VAD / Wake Word]
        VAD -->|Voice Stream| T_EDGE[Tailscale Mesh: 100.88.162.69]
        SCR[Screen & Notifications] -->|OCR / Accessibility| CTX_CAP[Context Extractor]
        CTX_CAP --> T_EDGE
        
        K_PROBE[Ring-0 eBPF Probes] -->|sys_execve / sys_connect| BPF_RELAY[bpf_relay]
        BPF_RELAY -->|WebSocket:8787| T_EDGE
        
        ACT_EXEC[Action Backend: Root / Frida / ADB] <---|Exec Payload| T_EDGE
    end

    subgraph "Workstation Host: RTX 3050 Compute Node"
        T_EDGE --> PERC[jarvis/perception: Whisper STT & Kokoro TTS]
        T_EDGE --> WARDEN[jarvis/telemetry: Warden & Anomaly Engine]
        
        PERC --> LOOP[jarvis/loop: Autonomous Planner & Replay Engine]
        WARDEN --> POLICY{jarvis/policy: Zero-Trust Safety Gate}
        
        LOOP --> POLICY
        
        POLICY -->|Tier-0 / Tier-1 Safe| AUDIT[jarvis/audit: Cryptographic Hash Chain]
        POLICY -->|Tier-2 / Tier-3 Gated| CONFIRM{User Confirmation / Invariants}
        
        CONFIRM -->|Approved| AUDIT
        AUDIT -->|Signed Payload| ACT_EXEC
        
        AUDIT --> DB[(Encrypted Temporal Store: demo.db)]
    end

    style K_PROBE fill:#e74c3c,stroke:#fff,stroke-width:2px,color:#fff
    style POLICY fill:#f39c12,stroke:#fff,stroke-width:2px,color:#fff
    style LOOP fill:#3498db,stroke:#fff,stroke-width:2px,color:#fff
    style AUDIT fill:#2ecc71,stroke:#fff,stroke-width:2px,color:#fff
```

---

## 🧩 Core Subsystems

### 1. `jarvis/perception` (Low-Latency Voice & Intent)
- **Silero ONNX VAD & Audio I/O:** Real-time voice activity detection with circular streaming ring buffers.
- **Wake Word Detection:** High-accuracy local activation ("Hey JARVIS").
- **Whisper STT (`Systran/faster-whisper-small`):** Host GPU accelerated batch/streaming transcription.
- **Kokoro-82M TTS (`kokoro-v1.0.onnx`):** Natural speech synthesis (<200ms time-to-first-audio).

### 2. `jarvis/context` (Mobile Temporal Memory)
- **Device Capture:** Real-time visual screen capture, accessibility node parsing, and notification harvesting.
- **Zero-Trust Privacy Filter:** On-the-fly regex/NLP scrubbing of PII, passwords, banking credentials, and private messages.
- **Encrypted Local Store (`demo.db`):** AES-256 encrypted SQLite temporal storage with blob offloading and TTL retention policies.

### 3. `jarvis/policy` (Zero-Trust Action Safety Gate)
- **Tier Classification:**
  - **Tier-0 (Safe Read):** Read clipboard, get battery, query system time (auto-approved).
  - **Tier-1 (Reversible UI):** Launch app, tap coordinates, scroll screen, media controls.
  - **Tier-2 (State Changing):** Send notification, write non-system settings, network connections.
  - **Tier-3 (Critical / Dangerous):** Shell execution, Magisk root commands, package install/uninstall, kernel map modifications (requires explicit interactive confirmation or active token).
- **Execution Backends:** Local ADB socket, Frida runtime hooks, and Direct Root Shell (`su`).

### 4. `jarvis/loop` (Autonomous Planning & Self-Improvement)
- **Execution Engine:** Dynamic plan synthesis with backtracking and invariant validation.
- **Continuous Trajectory Curation:** Harvests successful and failed action trajectories.
- **LoRA Fine-Tuning Factory (`train_lora.py`):** Trains lightweight adapter layers from operational traces for model alignment.

### 5. `jarvis/telemetry` (Warden & eBPF Telemetry)
- **In-Kernel Ingestion:** Collects ring-0 system events (`sys_execve`, `sys_connect`).
- **Feature Extraction & Attribution:** Computes event-level Kolmogorov complexity approximations and maps anomalies back to parent PIDs.

### 6. `jarvis/audit` (Cryptographic Audit & Hardening)
- **SHA-256 Merkle Ledger:** Strictly monotonic action log where any historical tampering invalidates the verification root.
- **Emergency Killswitch:** Instantaneous hardware-safe killswitch halting all automated routines and restoring safe default device states.
- **Deterministic Replay Harness:** Replay recorded incident traces to reproduce system states exactly.

---

## 🚀 Quickstart & Installation

### Prerequisites
- **Host OS:** Linux (x86_64 or aarch64), Python 3.10+
- **Compute:** NVIDIA GPU with CUDA support recommended (e.g., RTX 3050, 4GB+ VRAM)
- **Mobile Device:** OnePlus 9R (`lemonades`), LineageOS 20/21 or OxygenOS 13+, Magisk Root
- **Network:** Connected to the same Tailscale tailnet

### 1. Clone & Setup Python Environment
```bash
git clone https://github.com/Slattx7/JARVIS-OS-9R.git
cd JARVIS-OS-9R

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Verify System Test Suite
Run the full verification suite (282 comprehensive unit & integration tests):
```bash
pytest tests/ -v
```
*Expected result: 282 passed in ~20 seconds.*

### 3. Launching the OS
```bash
# Start JARVIS in autonomous daemon mode
jarvis-os --config config/jarvis.json --device 100.88.162.69:5555

# Or run the interactive policy gate CLI
python3 -m jarvis.policy.cli --interactive
```

---

## 🔒 Security & Safe Operation

> [!IMPORTANT]
> **Zero Bootloop Guarantee:** JARVIS-OS does **not** flash boot partitions or edit `/system` directly at runtime. All volatile mobile utilities operate inside `/data/local/tmp/` and the optional Magisk systemless overlay. A physical device reboot immediately reverts all transient kernel modifications.

> [!WARNING]
> **Network Screen Captures:** When orchestrating the OnePlus 9R over network ADB (`adb -s 100.x.y.z:5555`), avoid `exec-out screencap -p > local.png`. Always capture to on-device storage (`/sdcard/screencap.png`) before pulling to prevent truncated image corruption.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
