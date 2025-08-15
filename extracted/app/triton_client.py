import os, requests
class TritonClient:
    def __init__(self, url=None): self.url = url or os.getenv("TRITON_URL","http://localhost:8000")
    def health(self):
        try: return requests.get(self.url + "/v2/health/ready", timeout=2).status_code == 200
        except Exception: return False
