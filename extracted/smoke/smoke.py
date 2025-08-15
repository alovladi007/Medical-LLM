#!/usr/bin/env python3
import argparse, requests
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imaging', required=True)
    args = ap.parse_args()
    r = requests.get(f"{args.imaging}/v1/triton/health", timeout=5)
    print(r.status_code, r.text[:200])
if __name__ == '__main__':
    main()
