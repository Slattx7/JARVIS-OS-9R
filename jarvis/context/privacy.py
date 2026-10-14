"""Capture-time and ingest-time privacy policy. Stdlib-only (runs on device).

Defense in depth: the device applies this before anything is queued; the server applies it again
on ingest so a drifted device config cannot leak what the server's policy forbids.
"""
from __future__ import annotations

import copy
import fnmatch
import re
from typing import Any, Iterable

from .schema import BLOCKED_APP, TEXT_KEYS

# Starter lists. Package names are best-effort; verify against `pm list packages` on the device.
DEFAULT_BLOCK = (
    "com.bitwarden*", "com.x8bit.bitwarden*", "com.lastpass*", "com.agilebits*", "com.onepassword*",
    "proton.android.pass*", "*keepass*", "*authenticator*", "com.beemdevelopment.aegis*", "com.authy*",
    "com.twofasapp*", "org.fedorahosted.freeotp*",
)
DEFAULT_META = (
    "*bank*", "com.phonepe.app", "com.google.android.apps.nbu.paisa.user", "net.one97.paytm",
    "in.org.npci.upiapp", "com.snapwork.hdfc", "com.csam.icici.bank.imobile", "com.sbi.lotusintouch",
    "com.axis.mobile", "com.msf.kbank.mobile", "com.coinbase*", "com.binance*", "*wallet*",
)
META_KEYS = ("category", "channel", "importance", "initial")


def _pats(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(p.lower() for p in items)


def _match(pats: tuple[str, ...], app: str) -> bool:
    a = app.lower()
    return any(fnmatch.fnmatchcase(a, p) for p in pats)


class Denylist:
    """Per-package policy: 'block' (drop), 'meta' (keep metadata, strip content), or 'allow'.

    Precedence: user block > user meta > user allow > default block > default meta > allow.
    """

    def __init__(self, block: Iterable[str] = (), meta: Iterable[str] = (), allow: Iterable[str] = ()):
        self.block, self.meta, self.allow = _pats(block), _pats(meta), _pats(allow)
        self.d_block, self.d_meta = _pats(DEFAULT_BLOCK), _pats(DEFAULT_META)
        self._cache: dict[str, str] = {}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Denylist":
        return cls(cfg.get("block", ()), cfg.get("meta", ()), cfg.get("allow", ()))

    def mode(self, app: str) -> str:
        m = self._cache.get(app)
        if m is None:
            if app == BLOCKED_APP:
                m = "allow"
            elif _match(self.block, app):
                m = "block"
            elif _match(self.meta, app):
                m = "meta"
            elif _match(self.allow, app):
                m = "allow"
            elif _match(self.d_block, app):
                m = "block"
            elif _match(self.d_meta, app):
                m = "meta"
            else:
                m = "allow"
            self._cache[app] = m
        return m


_OTP_CTX = re.compile(
    r"\b(?:otp|one[\s-]?time|verification|verify|security code|login code|passcode|password|passwd|pin|cvv|cvc|2fa"
    r"|auth(?:entication)? code)\b", re.I)
_SHORT_NUM = re.compile(r"(?<![\w.])\d{4,8}(?!\w)")
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_AADHAAR = re.compile(r"(?<!\d)(?<!\d[ -])\d{4}[ -]\d{4}[ -]\d{4}(?!\d)(?![ -]\d)")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_SECRET = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})")
_URL = re.compile(r"https?://[^\s<>\"']+")


def _luhn(num: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(num)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_text(s: str) -> tuple[str, list[str]]:
    """Mask secrets, card numbers, national IDs, OTP-like codes and URL query strings."""
    flags: list[str] = []

    def flag(name: str, repl: str) -> str:
        if name not in flags:
            flags.append(name)
        return repl

    out = _SECRET.sub(lambda m: flag("secret", "[SECRET]"), s)

    def url(m: re.Match[str]) -> str:
        parts = re.split(r"[?#]", m.group(0), maxsplit=1)
        return flag("url-query", parts[0] + "?[Q]") if len(parts) == 2 else m.group(0)

    out = _URL.sub(url, out)

    def card(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn(digits):
            return flag("card", "[CARD]")
        return m.group(0)

    out = _CARD.sub(card, out)
    out = _AADHAAR.sub(lambda m: flag("id", "[ID]"), out)
    out = _PAN.sub(lambda m: flag("id", "[ID]"), out)
    if _OTP_CTX.search(out):
        out = _SHORT_NUM.sub(lambda m: flag("otp", "[OTP]"), out)
    return out, flags


def apply_policy(ev: dict[str, Any], deny: Denylist, *, redact: bool = True) -> dict[str, Any] | None:
    """Return a policy-conformant copy of the event, or None if it must be dropped entirely."""
    etype, app = ev["type"], ev["app"]
    out = copy.deepcopy(ev)
    data = out.setdefault("data", {})
    flags = set(out.get("flags") or [])
    mode = "allow" if etype in ("heartbeat", "screen") else deny.mode(app)

    if etype == "notification":
        if mode == "block":
            return None
        if mode == "meta":
            out["data"] = {k: v for k, v in data.items() if k in META_KEYS}
            flags.add("meta-only")
        elif redact:
            for k in TEXT_KEYS:
                if isinstance(data.get(k), str):
                    data[k], fl = redact_text(data[k])
                    flags.update(fl)
    elif etype == "app_switch":
        if mode == "block":
            out["app"], out["data"] = BLOCKED_APP, {}
            flags.add("blocked")
        else:
            frm = data.get("from")
            if isinstance(frm, str) and deny.mode(frm) == "block":
                data["from"] = BLOCKED_APP
            if mode == "meta":
                data.pop("activity", None)
    elif etype == "heartbeat":
        fg = data.get("fg")
        if isinstance(fg, str) and deny.mode(fg) == "block":
            data["fg"] = BLOCKED_APP
    out["flags"] = sorted(flags)
    return out
