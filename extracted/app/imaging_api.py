from fastapi import FastAPI
from triton_client import TritonClient
app = FastAPI()
@app.get("/v1/triton/health")
def health():
    cli = TritonClient()
    return {"available": bool(cli.url), "healthy": cli.health()}
