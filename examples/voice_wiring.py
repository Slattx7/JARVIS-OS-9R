"""Wiring sketch for the existing voice round-trip. Replace the three stubs with your real
wake/STT, local LLM call, and TTS. Everything else is the shipped API."""
from jarvis_actions.agent import Agent
from jarvis_actions.config import load_config
from jarvis_actions.confirm import VoiceConfirmer
from jarvis_actions.recall import HttpRecall
from jarvis_actions.registry import build_dispatcher, build_registry
from jarvis_actions.runner import Runner
from jarvis_actions.trace import CallbackSink


class TTS:                       # stub: your speaker
    def say(self, text): print("JARVIS:", text)

class Mic:                       # stub: your STT; must start listening only AFTER say() returns (echo!)
    def listen(self, timeout_s): return input("you> ") or None

def llm(system: str, user: str) -> str:   # stub: your local model, must return the JSON plan text
    raise NotImplementedError

cfg = load_config("config.example.toml")
probe = build_registry(cfg, Runner())
confirm = VoiceConfirmer(TTS(), Mic(), timeout_s=cfg.voice_timeout_s, strong_min_delay_s=cfg.strong_min_delay_s,
                         spec_lookup=probe.get)
recall = HttpRecall(cfg.recall_url) if cfg.recall_url else None
disp = build_dispatcher(cfg, confirm, on_alert=TTS().say, recall=recall,
                        extra_sinks=[CallbackSink(lambda ev: None)])      # -> jarvis_trace emitter
agent = Agent(disp, llm, TTS().say, recall=recall, devices=list(cfg.devices), default_device="nitro")

# per utterance:   agent.handle(transcript)
# proactive/bg:    agent.handle(suggestion, source=Source.DERIVED, use_recall=False)
# kill switch:     disp.lockdown()          # e.g. bind to a hardware key / phone shortcut
# undo:            disp.undo_last()
