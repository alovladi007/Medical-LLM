import os, requests
OPA_URL = os.getenv("OPA_URL","http://localhost:8181/v1/data/medagi/allow")
def allow(input_obj):
    try:
        r = requests.post(OPA_URL, json={"input": input_obj}, timeout=2)
        return r.json().get("result", False)
    except Exception:
        return False
