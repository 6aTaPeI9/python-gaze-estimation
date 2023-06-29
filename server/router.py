"""
    Роутинг запросов
"""
import uvicorn
import gaze_track
import json

from pydantic import BaseModel

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

class ImgProcess(BaseModel):
    img: str


origins = ['*']

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/calibrate", StaticFiles(directory="/home/python-gaze-estimation/server/brows_client", html = True), name="static")

@app.get("/ping")
def pong():
    """
        Тест эхо
    """
    return {"ping": "pong!"}

@app.post("/process/")
def process_img(data: ImgProcess):
    """
        Вычислние направления взгляда.
        На вход принимает jpg изображение закодированное
        в base64 строку

            frame: text - jpg изображение в base64 строке
            pwc1: int array - точка левого зрачка
            pwc2: int array - точка правого зрачка
            gaze1: int array - вектор для левого зрачка
            gaze2: int array - вектор для правого зрачка
    """
    data = data.img.replace('undefined', '')
    res = gaze_track.calc_gaze(data, w_frame=True)
    return json.dumps(res)

# @app.exception_handler(404)
# def handle_not_found(_,__):
#     return Stat

@app.websocket("/")
async def wsock(websocket: WebSocket):
    """
        Вычислние направления взгляда.
        На вход принимает jpg изображение закодированное
        в base64 строку.

        :return: json
            frame: text - jpg изображение в base64 строке
            pwc1: int array - точка левого зрачка
            pwc2: int array - точка правого зрачка
            gaze1: int array - вектор для левого зрачка
            gaze2: int array - вектор для правого зрачка
    """
    await websocket.accept()
    while True:
        res = await websocket.receive_text()
        try:
            print('На вход: ', res)
            res = json.loads(res)
            frame = res.get('frame')
            frame = frame.replace('undefined', '')
            result = gaze_track.calc_gaze(frame) or {}
            result.update(res.get('custom') or {})
        except Exception as ex:
            print(ex)
            result = {}
        result = await websocket.send_text(json.dumps(result))


if __name__ == '__main__':
    uvicorn.run("router:app", port=8485, log_level="info")
