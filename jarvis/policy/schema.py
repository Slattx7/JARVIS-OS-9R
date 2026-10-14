"""Minimal arg schema validation. Strict: unknown keys rejected, bool is not int."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

MAX_STR = 4096


@dataclass(frozen=True)
class Arg:
    type: type = str
    required: bool = True
    max_len: int = MAX_STR
    choices: Optional[tuple] = None
    min: Optional[float] = None
    max: Optional[float] = None
    pattern: Optional[str] = None     # regex, fullmatch
    doc: str = ""

    def describe(self) -> str:
        bits = [self.type.__name__]
        if self.choices: bits.append("one of " + "|".join(map(str, self.choices)))
        if self.min is not None or self.max is not None: bits.append(f"range [{self.min},{self.max}]")
        if self.pattern: bits.append(f"/{self.pattern}/")
        if not self.required: bits.append("optional")
        return " ".join(bits)


def validate(schema: dict, args: dict) -> list:
    import re
    errs = []
    if not isinstance(args, dict):
        return ["args must be an object"]
    for k in args:
        if k not in schema:
            errs.append(f"unknown arg '{k}'")
    for k, s in schema.items():
        if k not in args:
            if s.required: errs.append(f"missing arg '{k}'")
            continue
        v = args[k]
        if s.type is bool:
            if not isinstance(v, bool): errs.append(f"'{k}' must be bool"); continue
        elif s.type is int:
            if isinstance(v, bool) or not isinstance(v, int): errs.append(f"'{k}' must be int"); continue
        elif s.type is float:
            if isinstance(v, bool) or not isinstance(v, (int, float)): errs.append(f"'{k}' must be number"); continue
        elif s.type is str:
            if not isinstance(v, str): errs.append(f"'{k}' must be string"); continue
            if "\x00" in v: errs.append(f"'{k}' contains NUL"); continue
            if len(v) > s.max_len: errs.append(f"'{k}' too long ({len(v)}>{s.max_len})"); continue
            if s.pattern and not re.fullmatch(s.pattern, v): errs.append(f"'{k}' fails pattern"); continue
        else:
            errs.append(f"'{k}' has unsupported schema type"); continue
        if s.choices and v not in s.choices: errs.append(f"'{k}' not in choices")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if s.min is not None and v < s.min: errs.append(f"'{k}' < {s.min}")
            if s.max is not None and v > s.max: errs.append(f"'{k}' > {s.max}")
    return errs
