"""
    Роутинг запросов
"""
import uvicorn
import json
import sys
import os

from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


origins = ['*']

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CUR_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
app.mount("/admin", StaticFiles(directory=CUR_DIR + '\\', html = True), name="static")
DATA_DEVICES = {}

@app.get("/ping")
def pong():
    """
        Тест эхо
    """
    return {"ping": "pong!"}


@app.post("/save_process/")
def save_process(data: Dict[Any, Any]):
    if not data:
        return False

    print(data['device'])
    print(len(DATA_DEVICES))
    DATA_DEVICES[data['device']] = data


@app.post("/read_devices/")
def save_process():
    return json.dumps(DATA_DEVICES)

if __name__ == '__main__':
    uvicorn.run(app, port=8483, log_level="info")