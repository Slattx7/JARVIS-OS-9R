#!/usr/bin/env python3
import sys
import json
import time
import re
import threading

try:
    import websocket
except ImportError:
    print("Missing websocket-client", file=sys.stderr)
    sys.exit(1)

class BPFRelay:
    def __init__(self, driftnet_url):
        self.driftnet_url = driftnet_url
        self.ws = None
        self.ws_lock = threading.Lock()
        self._connect()

    def _connect(self):
        with self.ws_lock:
            try:
                self.ws = websocket.create_connection(self.driftnet_url, timeout=5)
                print(f"[bpf-relay] Connected to driftnetd at {self.driftnet_url}", file=sys.stderr)
            except Exception as e:
                print(f"[bpf-relay] Failed to connect: {e}", file=sys.stderr)
                self.ws = None

    def _send_event(self, event):
        payload = json.dumps(event)
        with self.ws_lock:
            if self.ws is None:
                self._connect()
            if self.ws is None:
                return
            try:
                self.ws.send(payload)
            except Exception as e:
                print(f"[bpf-relay] send failed, reconnecting: {e}", file=sys.stderr)
                self.ws = None

    def process_stdin(self):
        pattern = re.compile(r"PID:\s*(\d+)\s*\|\s*UID:\s*(\d+)\s*\|\s*COMM:\s*(.*?)\s*\|\s*FNAME:\s*(.*)")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            match = pattern.search(line)
            if match:
                pid, uid, comm, fname = match.groups()
                event = {
                    "device": "Android-Ring0",
                    "app": comm.strip(),
                    "kind": "sys_openat",
                    "detail": {
                        "pid": int(pid),
                        "uid": int(uid),
                        "fname": fname.strip()
                    }
                }
                self._send_event(event)

if __name__ == "__main__":
    relay = BPFRelay("ws://127.0.0.1:8787/ws/ingest")
    print("[bpf-relay] Listening for eBPF events on stdin...", file=sys.stderr)
    relay.process_stdin()
