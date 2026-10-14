"""Ed25519 for out-of-band approvals.

Backends: `cryptography` when installed, else a dependency-free RFC 8032 reference implementation, so the
*signer* (your phone in Termux) needs nothing beyond Python. The host only ever verifies (public key only).
The pure-Python path is not constant-time; acceptable for an occasional human-initiated signature on your own
device, and it is cross-checked against `cryptography` and the RFC 8032 vectors in the test-suite.
"""
from __future__ import annotations

import hashlib
import os

# ---------------- RFC 8032 section 6 reference implementation ----------------
_p = 2**255 - 19
_d = -121665 * pow(121666, _p - 2, _p) % _p
_q = 2**252 + 27742317777372353535851937790883648493


def _sha512(s: bytes) -> bytes:
    return hashlib.sha512(s).digest()


def _sha512_modq(s: bytes) -> int:
    return int.from_bytes(_sha512(s), "little") % _q


def _inv(x: int) -> int:
    return pow(x, _p - 2, _p)


def _add(P, Q):
    A, B = (P[1] - P[0]) * (Q[1] - Q[0]) % _p, (P[1] + P[0]) * (Q[1] + Q[0]) % _p
    C, D = 2 * P[3] * Q[3] * _d % _p, 2 * P[2] * Q[2] % _p
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F % _p, G * H % _p, F * G % _p, E * H % _p)


def _mul(s: int, P):
    Q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            Q = _add(Q, P)
        P = _add(P, P)
        s >>= 1
    return Q


def _eq(P, Q) -> bool:
    return (P[0] * Q[2] - Q[0] * P[2]) % _p == 0 and (P[1] * Q[2] - Q[1] * P[2]) % _p == 0


_sqrt_m1 = pow(2, (_p - 1) // 4, _p)


def _recover_x(y: int, sign: int):
    if y >= _p:
        return None
    x2 = (y * y - 1) * _inv(_d * y * y + 1)
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_p + 3) // 8, _p)
    if (x * x - x2) % _p != 0:
        x = x * _sqrt_m1 % _p
    if (x * x - x2) % _p != 0:
        return None
    if (x & 1) != sign:
        x = _p - x
    return x


_gy = 4 * _inv(5) % _p
_gx = _recover_x(_gy, 0)
_G = (_gx, _gy, 1, _gx * _gy % _p)


def _compress(P) -> bytes:
    zi = _inv(P[2])
    x, y = P[0] * zi % _p, P[1] * zi % _p
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(s: bytes):
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _p)


def _expand(seed: bytes):
    if len(seed) != 32:
        raise ValueError("Ed25519 seed must be 32 bytes")
    h = _sha512(seed)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def _py_public(seed: bytes) -> bytes:
    a, _ = _expand(seed)
    return _compress(_mul(a, _G))


def _py_sign(seed: bytes, msg: bytes) -> bytes:
    a, prefix = _expand(seed)
    A = _compress(_mul(a, _G))
    r = _sha512_modq(prefix + msg)
    Rs = _compress(_mul(r, _G))
    h = _sha512_modq(Rs + A + msg)
    return Rs + int.to_bytes((r + h * a) % _q, 32, "little")


def _py_verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    if len(pub) != 32 or len(sig) != 64:
        return False
    A = _decompress(pub)
    R = _decompress(sig[:32])
    if A is None or R is None:
        return False
    s = int.from_bytes(sig[32:], "little")
    if s >= _q:
        return False
    h = _sha512_modq(sig[:32] + pub + msg)
    return _eq(_mul(s, _G), _add(R, _mul(h, A)))


# ---------------- backend selection ----------------
try:  # pragma: no cover - exercised implicitly when installed
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

    _HAVE_CRYPTOGRAPHY = True
except Exception:  # noqa: BLE001
    _HAVE_CRYPTOGRAPHY = False


def backend() -> str:
    return "cryptography" if _HAVE_CRYPTOGRAPHY else "pure-python"


def generate_seed() -> bytes:
    return os.urandom(32)


def public_key(seed: bytes) -> bytes:
    if _HAVE_CRYPTOGRAPHY:
        k = Ed25519PrivateKey.from_private_bytes(seed)
        return k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return _py_public(seed)


def sign(seed: bytes, msg: bytes) -> bytes:
    if _HAVE_CRYPTOGRAPHY:
        return Ed25519PrivateKey.from_private_bytes(seed).sign(msg)
    return _py_sign(seed, msg)


def verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    if len(pub) != 32 or len(sig) != 64:
        return False
    if _HAVE_CRYPTOGRAPHY:
        try:
            Ed25519PublicKey.from_public_bytes(pub).verify(sig, msg)
            return True
        except (InvalidSignature, ValueError):
            return False
    return _py_verify(pub, msg, sig)
