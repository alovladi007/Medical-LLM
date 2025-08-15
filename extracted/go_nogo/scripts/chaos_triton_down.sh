#!/usr/bin/env bash
set -e
echo "Simulating Triton down..."
# This script assumes docker compose service name 'triton'
docker compose stop triton || true
sleep 5
echo "Run health check (should fall back to ORT/stub)..."
curl -s ${IMAGING:-http://localhost:8006}/v1/triton/health || true
echo "Bring Triton back..."
docker compose start triton || true
