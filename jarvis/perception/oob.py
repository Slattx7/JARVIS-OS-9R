"""Out-of-band approval channel for T3 actions.

Run the approver on a *different device* (your phone over Tailscale) that holds the approval key.
The approver reviews the exact action, signs (id, nonce, digest) locally, and POSTs only the signature.
The host holds just the public key. Bind to your tailscale0 address.

  GET  /pending?ts=<unix>&sig=<sign("list",ts)>  -> pending requests (readback, digest, nonce); ts within +-30 s
  POST /approve {"id":..., "token":...}          -> executes iff the signature verifies
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .approval import HmacVerifier, Verifier
from .broker import Broker


class OOBServer:
    MAX_BODY = 4096

    def __init__(self, broker: Broker, verifier: Verifier | bytes, bind: str = "127.0.0.1", port: int = 8765,
                 on_event=None, clock=time.time):
        if isinstance(verifier, (bytes, bytearray)):
            verifier = HmacVerifier(bytes(verifier))
        self.broker, self.verifier, self.on_event, self.clock = broker, verifier, on_event, clock
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence default stderr logging
                pass

            def _send(self, code: int, obj: dict) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                ts, sig = q.get("ts", [""])[0], q.get("sig", [""])[0]
                if u.path != "/pending" or not outer.verifier.verify_list(ts, sig, outer.clock()):
                    return self._send(403, {"error": "forbidden"})
                self._send(200, {"pending": [p for p in outer.broker.list_pending() if p["kind"] == "oob"]})

            def do_POST(self):
                if self.path != "/approve":
                    return self._send(404, {"error": "not found"})
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > outer.MAX_BODY:
                    return self._send(400, {"error": "bad length"})
                try:
                    obj = json.loads(self.rfile.read(n))
                    pid, token = str(obj["id"]), str(obj["token"])
                except (ValueError, KeyError, TypeError):
                    return self._send(400, {"error": "bad json"})
                out = outer.broker.approve_oob(pid, token)
                if outer.on_event:
                    outer.on_event("oob", {"id": pid, "status": out.status})
                self._send(200 if out.status == "executed" else 403,
                           {"status": out.status, "message": out.say or (out.result.output if out.result else "")})

        self._httpd = ThreadingHTTPServer((bind, port), H)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="oob")

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
