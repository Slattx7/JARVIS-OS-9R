# Project OMNI EDR

**Project OMNI** is a professional-grade Mobile Endpoint Detection and Response (EDR) daemon designed for advanced Android devices.

## Features
- **Ring-0 eBPF Probes**: Invisible, high-performance interception of `security_socket_connect` and `do_execve_file`.
- **Local AI Triage**: Employs an on-device Large Language Model (via `llama.cpp`) to actively evaluate novel system anomalies using thermal-aware "Lazy AI" architecture.
- **Ubuntu Edge Node**: Operates securely inside a native Ubuntu chroot, decoupling from Android's userland restrictions.

## Requirements
- **Architecture**: `arm64-v8a` (e.g., Snapdragon 870 / OnePlus 9R)
- **OS**: Android 10+ (API 29+) with Kernel `kprobes` and eBPF enabled (like crDroid 13).
- **Magisk**: Required for Late-Start Services and systemless deployment.

## Installation
Flash this ZIP via the Magisk Manager application. The interactive installer script will validate your architecture and OS version. Reboot to activate.

## Credits
Authored by **codeinedark77**.
