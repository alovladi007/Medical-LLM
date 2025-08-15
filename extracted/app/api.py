from fastapi import FastAPI
from exporter import ship
app = FastAPI()
@app.get("/v1/ops/siem/stats")
def stats():
    return {"ok": True}
@app.post("/v1/ops/siem/log")
def send_log(e: dict):
    code = ship(e)
    return {"code": code}
