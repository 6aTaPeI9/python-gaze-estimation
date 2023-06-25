"""
    Роутинг запросов
"""
import uvicorn
from fastapi import FastAPI, WebSocket
import gaze_track
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class ImgProcess(BaseModel):
    img: str


origins = ['*']

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["DELETE", "GET", "POST", "PUT"],
    allow_headers=["*"],
)

@app.get("/ping")
def pong():
    """
        Тест эхо
    """
    return {"ping": "pong!"}

@app.post("/process/")
def process_img(data: ImgProcess):
    """
    """
    data = data.img.replace('undefined', '')
    res = gaze_track.calc_gaze(data)
    # print()
    print('Обработали: ', res.__sizeof__())

    return res

@app.websocket("/")
async def wsock(websocket: WebSocket):
    """
        Вычислние направления взгляда
    """
    await websocket.accept()
    while True:
        res = await websocket.receive_bytes()
        print('Получили', res.__sizeof__())
        res = gaze_track.calc_gaze(res)
        res = await websocket.send_bytes(res)


if __name__ == '__main__':
    uvicorn.run("router:app", port=8486, log_level="info")
