#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/go_nogo"
python3 bootstrap.py --profile dev   --imaging ${IMAGING:-http://localhost:8006}   --eval ${EVAL:-http://localhost:8005}   --anchor ${ANCHOR:-http://localhost:8007}   --modelcards ${MODELCARDS:-http://localhost:8008}   --ops ${OPS:-http://localhost:8010}
echo "Pilot bootstrap complete. See go_nogo/pilot_bootstrap_report.json"
