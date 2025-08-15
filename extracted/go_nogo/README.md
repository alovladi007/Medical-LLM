# Pilot Go/No‑Go Readiness Sweep

This package includes:
- `bootstrap.py` — validates env, spins up compose, runs health checks and optional smoke tests, writes a JSON report.
- `alerts/siem_policies.json` — logs‑based alert templates (error rate, p95 latency, Triton down).
- `scripts/k6/*.js` — k6 load tests for CXR/EKG/DICOM endpoints.
- `scripts/chaos_triton_down.sh` — basic chaos check to verify fallback behavior.
- `docs/DATA_RETENTION.md` — retention guidance for logs, PHI images, and eval artifacts.
- `ui/StatusWidget.tsx` — tiny UI badge you can drop into your app header.

## Quick start
```bash
python3 bootstrap.py --profile dev   --imaging http://localhost:8006   --eval http://localhost:8005   --anchor http://localhost:8007   --modelcards http://localhost:8008   --ops http://localhost:8010
```

## Load testing
```bash
k6 run scripts/k6/cxr.js   -e IMAGING=http://localhost:8006
k6 run scripts/k6/ekg.js   -e EKG=http://localhost:8016
k6 run scripts/k6/dicom.js -e IMAGING=http://localhost:8006
```

## Chaos
```bash
IMAGING=http://localhost:8006 bash scripts/chaos_triton_down.sh
```

