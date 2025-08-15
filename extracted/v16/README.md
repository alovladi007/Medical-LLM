# Med‑AGI (v16) — Pilot‑Ready Bundle

Includes:
- smoke/smoke.py — Pre‑pilot smoke test (JWT‑aware).
- RUNBOOK.md — Deployment, rollback, safety controls, incident flow.
- PILOT_CHECKLIST.md — Go/No‑Go preflight.
- scripts/publish_modelcards.py — Pushes latest eval model card JSON into UI’s public dir.
- tests/test_endpoints.py — Minimal endpoint tests.
- .github/workflows/smoke.yml — CI job running the tests.
