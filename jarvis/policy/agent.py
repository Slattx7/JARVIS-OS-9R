"""Agent loop: owner utterance -> (recall) -> LLM plan -> policy-gated execution -> speak.

The LLM and speaker are injected callables, so this plugs into the existing voice round-trip:
    agent = Agent(disp, llm=lambda system, user: my_local_llm(system, user), speak=tts.say)
    agent.handle(transcript)
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from .core import Dispatcher, Source
from .planner import PlanError, parse_plan, run_plan, tool_prompt
from .recall import RecallClient, snippets

MAX_TURNS = 4


@dataclass
class AgentResult:
    say: str = ""
    results: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    turns: int = 0


def _untrusted(label: str, text: str) -> str:
    text = text.replace("</untrusted>", "")        # no tag escape
    return f"<untrusted source='{label}'>\n{text[:4000]}\n</untrusted>"


class Agent:
    def __init__(self, disp: Dispatcher, llm: Callable[[str, str], str], speak: Optional[Callable[[str], None]] = None,
                 recall: Optional[RecallClient] = None, devices=None, default_device="nitro", recall_k=3):
        self.d, self.llm, self.speak, self.recall = disp, llm, speak, recall
        self.devices = list(devices or ["nitro", "9r", "14pro"])
        self.default_device, self.k = default_device, recall_k
        self.system = tool_prompt(disp.reg, self.devices)

    def _say(self, res: AgentResult, text: str):
        if text:
            res.say = (res.say + " " + text).strip()
            if self.speak: self.speak(text)

    def handle(self, utterance: str, *, source: Source = Source.OWNER, use_recall: bool = True) -> AgentResult:
        """source=OWNER for live voice; DERIVED for background/proactive suggestions."""
        res = AgentResult()
        if source == Source.OWNER and self.d.ledger is not None:
            self.d.ledger.trust(utterance)
        ctx = ""
        if use_recall and self.recall is not None:
            try:
                sn = snippets(self.recall.query(utterance, self.k))
            except Exception as e:
                sn = []; res.errors.append(f"recall: {e!r}")
            for s in sn:
                if self.d.ledger is not None: self.d.ledger.add(s, "recall.query")
            if sn: ctx = "\n".join(_untrusted("recall", s) for s in sn)
        user = f"Owner said: {utterance}\n" + (f"Context:\n{ctx}\n" if ctx else "")
        for turn in range(1, MAX_TURNS + 1):
            res.turns = turn
            try:
                plan = parse_plan(self.llm(self.system, user))
            except PlanError as e:
                res.errors.append(f"plan: {e}"); self._say(res, "I couldn't form a valid plan."); return res
            self._say(res, plan.say)
            rs = run_plan(self.d, plan, source, self.default_device)
            res.results += rs
            for r in rs:
                if r.get("denied"):
                    self._say(res, f"{r['name']} was blocked: {'; '.join(r.get('reasons', []))[:160]}")
            if plan.done or not rs or any(r.get("denied") for r in rs):
                return res
            feedback = "\n".join(_untrusted(r["name"], json.dumps(r.get("result", r.get("error", "")), default=str)) for r in rs)
            user = f"Owner said: {utterance}\nTool results:\n{feedback}\n"
        res.errors.append("max turns reached")
        return res
