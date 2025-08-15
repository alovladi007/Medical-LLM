import os, time, json, requests
SIEM_URL = os.getenv("SIEM_URL","http://localhost:8088/ingest")
def ship(event:dict):
    try:
        r = requests.post(SIEM_URL, json=event, timeout=2)
        return r.status_code
    except Exception:
        return 0
