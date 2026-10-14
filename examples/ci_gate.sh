#!/bin/sh
# Pre-merge / pre-deploy gate for a change to the model, prompt, policy, orchestrator or recall layer.
# Exit codes: 0 ok | 1 regression / lint / invariant failure | 2 SAFETY failure (a gate got looser, an action ran
# ungated, a new risky proposal appeared). Any non-zero exit blocks the change.
#
#   STORE=~/jarvis/traces.db AGENT=yourpkg.orchestrator:agent DEPLOYED_POLICY=policy.deployed.json ./examples/ci_gate.sh
set -eu
STORE="${STORE:-$HOME/.jarvis/traces.db}"
AGENT="${AGENT:?set AGENT=module:function (fn(rt, inputs))}"
DEPLOYED_POLICY="${DEPLOYED_POLICY:-}"
POLICY="${POLICY:-examples/policy.json}"
PY="${PYTHON:-python3}"

echo "== unit tests";                $PY -m unittest discover -s tests -t .
echo "== store integrity";           $PY -m jarvis_trace verify --store "$STORE"
echo "== policy lint";               $PY -m jarvis_trace policy-lint "$POLICY" --strict
echo "== policy cases";              $PY -m jarvis_trace policy-test "$POLICY" examples/policy_cases.json
if [ -n "$DEPLOYED_POLICY" ]; then
  echo "== policy diff vs deployed (every recorded check)"
  $PY -m jarvis_trace policy-diff --old "$DEPLOYED_POLICY" --new "$POLICY" --use-store --store "$STORE"
fi
echo "== replay golden set";         $PY -m jarvis_trace replay --agent "$AGENT" --gate examples/gate.json \
                                        --store "$STORE" --html replay.html --out replay.md --json replay.json
echo "== live invariants (last 7d)"; $PY -m jarvis_trace check --since 7d --store "$STORE"
echo "== drift (14d..7d vs 7d)";     $PY -m jarvis_trace drift --store "$STORE" || echo "(drift alerts above: review, not blocking)"
echo "gate passed"
