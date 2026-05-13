from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from service.api.upload_api import router as upload_router
from service.api.ask_api import router as ask_router

app = FastAPI(title="AI Risk Engine")

app.include_router(upload_router)
app.include_router(ask_router)

static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join(static_dir, "index.html"))