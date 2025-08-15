#!/usr/bin/env python3
import os, sys, subprocess, json, time, argparse, shutil, urllib.request

REQ_ENVS = [
  "OIDC_ISSUER","OIDC_AUDIENCE","OIDC_JWKS_URL",
  "OPA_URL","SIEM_URL","SIEM_MODE",
  "DICOMWEB_BASE","TRITON_URL"
]

def check_env():
  missing = [e for e in REQ_ENVS if not os.environ.get(e)]
  return missing

def http_get(url, timeout=5):
  try:
    with urllib.request.urlopen(url, timeout=timeout) as r:
      return r.status, r.read().decode("utf-8", "ignore")[:500]
  except Exception as e:
    return 0, str(e)

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--profile", choices=["dev","prod"], default="dev")
  ap.add_argument("--imaging", default="http://localhost:8006")
  ap.add_argument("--eval", default="http://localhost:8005")
  ap.add_argument("--anchor", default="http://localhost:8007")
  ap.add_argument("--modelcards", default="http://localhost:8008")
  ap.add_argument("--ops", default="http://localhost:8010")
  ap.add_argument("--compose", default="docker compose")
  ap.add_argument("--smoke", default="smoke/smoke.py")
  args = ap.parse_args()

  print("== Pilot bootstrap ==")
  missing = check_env()
  if missing:
    print("! Missing env vars:", ", ".join(missing))
  else:
    print("✓ Env vars present")

  # Bring up compose profile (non-blocking up -d)
  cmd = f"{args.compose} --profile {args.profile} up -d"
  print(">>", cmd); rc = os.system(cmd)
  if rc != 0: print("! compose up failed (continuing to checks)")

  # Health checks
  checks = {
    "imaging_triton": f"{args.imaging}/v1/triton/health",
    "eval_summary": f"{args.eval}/v1/eval/summary",
    "anchor_search": f"{args.anchor}/v1/anchor_search",
    "modelcards_list": f"{args.modelcards}/v1/modelcards/list",
    "ops_siem": f"{args.ops}/v1/ops/siem/stats",
  }
  results = {}
  for name, url in checks.items():
    code, body = http_get(url)
    results[name] = {"status": code, "body": body}
    print(f"{name}: {code}")

  # Smoke test (if present)
  if os.path.exists(args.smoke):
    print("Running smoke test...")
    rc = os.system(f"python3 {args.smoke} --imaging {args.imaging} --eval {args.eval} --anchor {args.anchor} --modelcards {args.modelcards} --ops {args.ops}")
    print("Smoke exit:", rc)

  # Keycloak seed (dev only, optional placeholder)
  if args.profile == "dev":
    print("Dev profile: ensure Keycloak seeded (skip if not used).")

  # Write report
  Path("pilot_bootstrap_report.json").write_text(json.dumps({
    "env_missing": missing,
    "checks": results,
    "profile": args.profile
  }, indent=2))
  print("Report -> pilot_bootstrap_report.json")

if __name__ == "__main__":
  main()
