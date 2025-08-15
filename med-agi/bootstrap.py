#!/usr/bin/env python3
"""
Med-AGI Bootstrap Script
Validates environment, starts services, runs health checks, and generates reports.
"""

import os
import sys
import subprocess
import json
import time
import argparse
import shutil
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Required environment variables
REQ_ENVS = [
    "OIDC_ISSUER", "OIDC_AUDIENCE", "OIDC_JWKS_URL",
    "OPA_URL", "SIEM_URL", "SIEM_MODE",
    "DICOMWEB_BASE", "TRITON_URL"
]

# Service configuration
SERVICES = {
    "imaging": {
        "port": 8006,
        "health": "/v1/triton/health",
        "name": "Imaging Service"
    },
    "ekg": {
        "port": 8016,
        "health": "/v1/ekg/health",
        "name": "EKG Service"
    },
    "eval": {
        "port": 8005,
        "health": "/v1/eval/health",
        "summary": "/v1/eval/summary",
        "name": "Evaluation Service"
    },
    "anchor": {
        "port": 8007,
        "health": "/v1/anchor/health",
        "search": "/v1/anchor_search",
        "name": "Anchor Service"
    },
    "modelcards": {
        "port": 8008,
        "health": "/v1/modelcards/health",
        "list": "/v1/modelcards/list",
        "name": "ModelCards Service"
    },
    "ops": {
        "port": 8010,
        "health": "/v1/ops/health",
        "siem": "/v1/ops/siem/stats",
        "name": "Operations Service"
    }
}


def check_env() -> List[str]:
    """Check for required environment variables."""
    missing = [e for e in REQ_ENVS if not os.environ.get(e)]
    return missing


def http_get(url: str, timeout: int = 5) -> Tuple[int, str]:
    """Perform HTTP GET request with timeout."""
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Med-AGI-Bootstrap/1.0')
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", "ignore")
            return response.status, body[:500]
    except urllib.error.HTTPError as e:
        return e.code, str(e)
    except urllib.error.URLError as e:
        return 0, f"Connection error: {e.reason}"
    except Exception as e:
        return 0, str(e)


def check_docker() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def check_compose() -> str:
    """Determine the correct docker-compose command."""
    commands = ["docker compose", "docker-compose"]
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd.split() + ["version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return cmd
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    return ""


def run_command(cmd: str, check: bool = True) -> int:
    """Run a shell command and return exit code."""
    print(f">> {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,
            text=True
        )
        if check and result.returncode != 0:
            print(f"! Command failed with exit code {result.returncode}")
        return result.returncode
    except subprocess.SubprocessError as e:
        print(f"! Command error: {e}")
        return 1


def wait_for_services(services: Dict[str, str], max_wait: int = 60) -> Dict[str, Dict]:
    """Wait for services to become healthy."""
    results = {}
    start_time = time.time()
    
    print(f"Waiting for services to become healthy (max {max_wait}s)...")
    
    for name, url in services.items():
        service_healthy = False
        attempts = 0
        
        while time.time() - start_time < max_wait and not service_healthy:
            attempts += 1
            code, body = http_get(url, timeout=3)
            
            if code == 200:
                service_healthy = True
                results[name] = {
                    "status": code,
                    "body": body,
                    "healthy": True,
                    "attempts": attempts
                }
                print(f"✓ {name}: healthy after {attempts} attempts")
            else:
                time.sleep(2)
        
        if not service_healthy:
            code, body = http_get(url, timeout=3)
            results[name] = {
                "status": code,
                "body": body,
                "healthy": False,
                "attempts": attempts
            }
            print(f"✗ {name}: not healthy after {attempts} attempts (status: {code})")
    
    return results


def run_smoke_tests(args: argparse.Namespace) -> int:
    """Run smoke tests if available."""
    smoke_script = Path(args.smoke)
    
    if not smoke_script.exists():
        print(f"! Smoke test script not found: {smoke_script}")
        return 1
    
    print("\n== Running smoke tests ==")
    
    cmd = [
        sys.executable, str(smoke_script),
        "--imaging", args.imaging,
        "--eval", args.eval,
        "--anchor", args.anchor,
        "--modelcards", args.modelcards,
        "--ops", args.ops
    ]
    
    if args.ekg:
        cmd.extend(["--ekg", args.ekg])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        return result.returncode
    except subprocess.TimeoutExpired:
        print("! Smoke tests timed out")
        return 1
    except subprocess.SubprocessError as e:
        print(f"! Smoke test error: {e}")
        return 1


def generate_report(
    profile: str,
    env_missing: List[str],
    health_results: Dict,
    smoke_result: Optional[int],
    start_time: float
) -> Path:
    """Generate a comprehensive bootstrap report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "profile": profile,
        "duration_seconds": round(time.time() - start_time, 2),
        "environment": {
            "missing_vars": env_missing,
            "all_present": len(env_missing) == 0
        },
        "health_checks": health_results,
        "smoke_tests": {
            "executed": smoke_result is not None,
            "exit_code": smoke_result,
            "passed": smoke_result == 0 if smoke_result is not None else None
        },
        "summary": {
            "total_services": len(health_results),
            "healthy_services": sum(1 for r in health_results.values() if r.get("healthy")),
            "ready": len(env_missing) == 0 and all(
                r.get("healthy") for r in health_results.values()
            )
        }
    }
    
    report_path = Path("pilot_bootstrap_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    
    # Also generate a human-readable summary
    summary_path = Path("pilot_bootstrap_summary.txt")
    with summary_path.open("w") as f:
        f.write(f"Med-AGI Bootstrap Report\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Timestamp: {report['timestamp']}\n")
        f.write(f"Profile: {report['profile']}\n")
        f.write(f"Duration: {report['duration_seconds']}s\n\n")
        
        f.write(f"Environment Check:\n")
        if report['environment']['all_present']:
            f.write(f"  ✓ All required environment variables present\n")
        else:
            f.write(f"  ✗ Missing variables: {', '.join(report['environment']['missing_vars'])}\n")
        
        f.write(f"\nService Health:\n")
        for name, result in health_results.items():
            status = "✓" if result.get("healthy") else "✗"
            f.write(f"  {status} {name}: {result.get('status', 'unknown')}\n")
        
        f.write(f"\nSmoke Tests:\n")
        if report['smoke_tests']['executed']:
            status = "✓ Passed" if report['smoke_tests']['passed'] else "✗ Failed"
            f.write(f"  {status} (exit code: {report['smoke_tests']['exit_code']})\n")
        else:
            f.write(f"  - Not executed\n")
        
        f.write(f"\nOverall Status: ")
        if report['summary']['ready']:
            f.write(f"✓ READY FOR PILOT\n")
        else:
            f.write(f"✗ NOT READY (see issues above)\n")
    
    return report_path


def main():
    """Main bootstrap orchestration."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description="Med-AGI System Bootstrap",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--profile",
        choices=["dev", "prod", "test"],
        default="dev",
        help="Deployment profile"
    )
    
    # Service endpoints
    parser.add_argument("--imaging", default="http://localhost:8006", help="Imaging service URL")
    parser.add_argument("--ekg", default="http://localhost:8016", help="EKG service URL")
    parser.add_argument("--eval", default="http://localhost:8005", help="Eval service URL")
    parser.add_argument("--anchor", default="http://localhost:8007", help="Anchor service URL")
    parser.add_argument("--modelcards", default="http://localhost:8008", help="ModelCards service URL")
    parser.add_argument("--ops", default="http://localhost:8010", help="Ops service URL")
    
    # Options
    parser.add_argument("--compose", default="", help="Docker Compose command")
    parser.add_argument("--smoke", default="smoke/smoke.py", help="Smoke test script path")
    parser.add_argument("--skip-compose", action="store_true", help="Skip Docker Compose startup")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke tests")
    parser.add_argument("--wait", type=int, default=60, help="Max wait time for services (seconds)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Med-AGI Bootstrap - Profile: {args.profile}")
    print("=" * 60)
    
    # 1. Check environment variables
    print("\n== Environment Check ==")
    missing_env = check_env()
    if missing_env:
        print(f"⚠ Missing environment variables: {', '.join(missing_env)}")
        if args.profile == "prod":
            print("! Production profile requires all environment variables")
            print("! Continuing anyway for validation...")
    else:
        print("✓ All required environment variables present")
    
    # 2. Check Docker
    print("\n== Docker Check ==")
    if not check_docker():
        print("✗ Docker is not available or not running")
        if not args.skip_compose:
            print("! Cannot proceed without Docker")
            sys.exit(1)
    else:
        print("✓ Docker is available")
    
    # 3. Start services with Docker Compose
    if not args.skip_compose:
        print("\n== Starting Services ==")
        
        # Determine compose command
        compose_cmd = args.compose or check_compose()
        if not compose_cmd:
            print("✗ Docker Compose not found")
            sys.exit(1)
        
        print(f"Using: {compose_cmd}")
        
        # Create docker-compose.yml if it doesn't exist
        compose_file = Path("docker-compose.yml")
        if not compose_file.exists():
            print("! docker-compose.yml not found, skipping compose startup")
        else:
            # Start services
            cmd = f"{compose_cmd} --profile {args.profile} up -d"
            rc = run_command(cmd)
            
            if rc != 0:
                print("⚠ Docker Compose startup had issues")
            else:
                print("✓ Services starting...")
                time.sleep(5)  # Give services time to initialize
    
    # 4. Health checks
    print("\n== Health Checks ==")
    health_endpoints = {
        "imaging_health": f"{args.imaging}/v1/triton/health",
        "ekg_health": f"{args.ekg}/v1/ekg/health",
        "eval_summary": f"{args.eval}/v1/eval/summary",
        "anchor_search": f"{args.anchor}/v1/anchor_search",
        "modelcards_list": f"{args.modelcards}/v1/modelcards/list",
        "ops_siem": f"{args.ops}/v1/ops/siem/stats",
    }
    
    health_results = wait_for_services(health_endpoints, max_wait=args.wait)
    
    # 5. Smoke tests
    smoke_result = None
    if not args.skip_smoke:
        smoke_result = run_smoke_tests(args)
    
    # 6. Keycloak seed for dev profile
    if args.profile == "dev":
        print("\n== Dev Profile Setup ==")
        print("Note: Ensure Keycloak is seeded if using OIDC authentication")
    
    # 7. Generate report
    print("\n== Generating Report ==")
    report_path = generate_report(
        args.profile,
        missing_env,
        health_results,
        smoke_result,
        start_time
    )
    
    print(f"✓ Report saved to: {report_path}")
    print(f"✓ Summary saved to: pilot_bootstrap_summary.txt")
    
    # 8. Final status
    print("\n" + "=" * 60)
    ready = (
        len(missing_env) == 0 and
        all(r.get("healthy") for r in health_results.values()) and
        (smoke_result == 0 if smoke_result is not None else True)
    )
    
    if ready:
        print("✓ SYSTEM READY FOR PILOT")
        print("=" * 60)
        return 0
    else:
        print("✗ SYSTEM NOT READY - CHECK REPORT FOR ISSUES")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())