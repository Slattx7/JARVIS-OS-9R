"""Approval schemes for T3 (out-of-band) requests.

ed25519 (default): the phone holds the private seed and signs; the host holds only the public key, so a
                   compromised host cannot mint approvals. Message = domain-separated (id, nonce, action digest).
hmac (legacy)    : shared secret; stops prompt injection / LLM / voice spoofing but not host compromise.

Key files are single-line, typed: `ed25519-seed:<hex>` (private, mode 600, lives on the phone),
`ed25519-pub:<hex>` (host), `hmac:<hex>`. A bare hex line is read as a legacy HMAC secret.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path

from . import crypto

APPROVAL_DOMAIN = b"jarvis-approval-v1\n"
LIST_DOMAIN = b"jarvis-list-v1\n"
LIST_WINDOW_S = 30.0


def approval_message(pending_id: str, nonce: str, digest: str) -> bytes:
    return APPROVAL_DOMAIN + f"{pending_id}\n{nonce}\n{digest}".encode()


def _hmac_token(secret: bytes, digest: str, nonce: str, pending_id: str) -> str:
    return hmac.new(secret, f"{digest}:{nonce}:{pending_id}".encode(), hashlib.sha256).hexdigest()


# ---------------- verifiers (host side) ----------------
class Verifier:
    scheme = "none"
    configured = False

    def verify_approval(self, pending_id: str, nonce: str, digest: str, token: str) -> bool:
        return False

    def verify_list(self, ts: str, sig: str, now: float) -> bool:
        return False


class NullVerifier(Verifier):
    """No approver key configured: fail closed. T3 requests are refused up front."""


def _fresh(ts: str, now: float) -> bool:
    try:
        return abs(now - float(ts)) <= LIST_WINDOW_S
    except ValueError:
        return False


class Ed25519Verifier(Verifier):
    scheme, configured = "ed25519", True

    def __init__(self, pub: bytes):
        if len(pub) != 32:
            raise ValueError("Ed25519 public key must be 32 bytes")
        self.pub = pub

    def _check(self, msg: bytes, sig_hex: str) -> bool:
        try:
            sig = bytes.fromhex(str(sig_hex))
        except ValueError:
            return False
        return crypto.verify(self.pub, msg, sig)

    def verify_approval(self, pending_id, nonce, digest, token) -> bool:
        return self._check(approval_message(pending_id, nonce, digest), token)

    def verify_list(self, ts, sig, now) -> bool:
        return _fresh(ts, now) and self._check(LIST_DOMAIN + str(ts).encode(), sig)


class HmacVerifier(Verifier):
    scheme, configured = "hmac", True

    def __init__(self, secret: bytes):
        self.secret = secret

    def verify_approval(self, pending_id, nonce, digest, token) -> bool:
        return hmac.compare_digest(_hmac_token(self.secret, digest, nonce, pending_id), str(token))

    def verify_list(self, ts, sig, now) -> bool:
        exp = hmac.new(self.secret, b"list:" + str(ts).encode(), hashlib.sha256).hexdigest()
        return _fresh(ts, now) and hmac.compare_digest(exp, str(sig))


# ---------------- signers (phone side) ----------------
class Signer:
    def sign_approval(self, pending_id: str, nonce: str, digest: str) -> str:
        raise NotImplementedError

    def sign_list(self, ts: str) -> str:
        raise NotImplementedError


class Ed25519Signer(Signer):
    def __init__(self, seed: bytes):
        self.seed = seed

    def sign_approval(self, pending_id, nonce, digest) -> str:
        return crypto.sign(self.seed, approval_message(pending_id, nonce, digest)).hex()

    def sign_list(self, ts) -> str:
        return crypto.sign(self.seed, LIST_DOMAIN + str(ts).encode()).hex()


class HmacSigner(Signer):
    def __init__(self, secret: bytes):
        self.secret = secret

    def sign_approval(self, pending_id, nonce, digest) -> str:
        return _hmac_token(self.secret, digest, nonce, pending_id)

    def sign_list(self, ts) -> str:
        return hmac.new(self.secret, b"list:" + str(ts).encode(), hashlib.sha256).hexdigest()


# ---------------- key files ----------------
def _read_typed(path: str | os.PathLike) -> tuple[str, bytes]:
    text = Path(os.path.expanduser(str(path))).read_text().strip()
    kind, sep, hexs = text.partition(":")
    if not sep:
        kind, hexs = "hmac", text
    return kind, bytes.fromhex(hexs)


def _write_private(path: Path, line: str, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if force else os.O_EXCL)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(line + "\n")
    os.chmod(path, 0o600)


def generate_keypair(private_path: str, public_path: str, force: bool = False) -> bytes:
    seed = crypto.generate_seed()
    priv, pub = Path(os.path.expanduser(private_path)), Path(os.path.expanduser(public_path))
    _write_private(priv, "ed25519-seed:" + seed.hex(), force)
    pub.parent.mkdir(parents=True, exist_ok=True)
    pub.write_text("ed25519-pub:" + crypto.public_key(seed).hex() + "\n")
    return seed


def generate_hmac_secret(path: str, force: bool = False) -> bytes:
    key = os.urandom(32)
    _write_private(Path(os.path.expanduser(path)), "hmac:" + key.hex(), force)
    return key


def load_signer(path: str | os.PathLike) -> Signer:
    kind, raw = _read_typed(path)
    if kind == "ed25519-seed":
        return Ed25519Signer(raw)
    if kind == "hmac":
        return HmacSigner(raw)
    raise ValueError(f"{path}: not a signing key (type {kind!r}); the approver needs the private seed, not the .pub")


def load_verifier(cfg, log=print) -> Verifier:
    b = cfg.broker
    if b.approval_scheme == "hmac":
        p = Path(os.path.expanduser(b.secret_path))
        if not p.exists():
            generate_hmac_secret(str(p))
        kind, raw = _read_typed(p)
        return HmacVerifier(raw)
    if b.approval_scheme != "ed25519":
        raise ValueError(f"unknown broker.approval_scheme {b.approval_scheme!r}")
    p = Path(os.path.expanduser(b.approver_pub_path))
    if not p.exists():
        log(f"WARNING: {p} missing: T3 (root) actions are DISABLED. Run `jarvis keygen` on your phone and copy the .pub here.")
        return NullVerifier()
    kind, raw = _read_typed(p)
    if kind != "ed25519-pub":
        raise ValueError(f"{p}: expected an ed25519 public key, found {kind!r}")
    return Ed25519Verifier(raw)


@dataclass
class ListAuth:
    ts: str
    sig: str


def make_list_auth(signer: Signer, now: float) -> ListAuth:
    ts = f"{now:.0f}"
    return ListAuth(ts, signer.sign_list(ts))
