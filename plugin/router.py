"""
    Роутинг запросов
"""
import uvicorn
from pydantic import BaseModel

import time
import json
import os
import multiprocessing

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from traker import monitor_calibrate, ray_intersect
from websockets.sync.client import connect



class FramesData(BaseModel):
    frame_data: dict

origins = ['*']

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FL_DIR_PATH = os.getenv('APPDATA') + '\\WebTracker\\'
FL_PATH = FL_DIR_PATH + 'status.txt'
os.makedirs(FL_DIR_PATH, exist_ok=True)


def sub_process():
    def get_image():
        _, frame = CAM.read()
        _, image = cv2.imencode('.jpg', frame)
        image = base64.b64encode(image).decode()
        return image

    import cv2
    import base64
    import requests

    CAM = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    with open(FL_PATH, 'r+') as fl:
        dt = fl.readline()

    dt = json.loads(dt)

    while True:
        try:
            with connect("ws://127.0.0.1:8485/") as websocket:
                for i in range(10):
                    time.sleep(0.2)

                    try:
                        requests.get('http://localhost:8484/ping/')
                    except Exception as ex:
                        break

                    try:
                        print('1')
                        websocket.send(get_image())
                        res = websocket.recv()
                        res = json.loads(res)
                        print(res)
                        print(dt)
                        print('2')
                        status = ray_intersect(res['pwc1'], res['pwc2'], res['gaze1'], res['gaze2'], dt['left_mon'], dt['right_mon'])
                        print(status)
                        print('3')
                        with open(FL_PATH, 'w+') as fl:
                            fl.write(json.dumps({
                                'face': res.get('frame'),
                                'status': status,
                                **dt
                            }))
                        print('4')
                    except Exception as ex:
                        if '-215:Assertion failed' in str(ex):
                            CAM = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                        print(ex)
        except Exception as ex:
            print(ex)

@app.get("/ping")
def pong():
    """
        Тест эхо
    """
    proc = multiprocessing.Process(target=sub_process, name='test', args=tuple())
    proc.daemon = True
    proc.start()
    return {"ping": "pong!"}

@app.post("/run_monitor/")
async def run_monitor(request: Request):
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
    dt = await request.json()
    dt_list = {}

    for k, v_list in dt.get('frame_data').items():
        if not dt_list.get(k):
            dt_list[k] = []

        for v in v_list:
            dt_list[k].append([v['pwc1'], v['pwc2'], v['gaze1'], v['gaze2']])

    res = monitor_calibrate(dt_list)

    if not res:
        raise ValueError('Не удалось расчитать откалибровать.')

    with open(FL_PATH, 'w+') as fl:
        fl.write(json.dumps({
            'left_mon': res[0],
            'right_mon': res[1]
        }))

    proc = multiprocessing.Process(target=sub_process, name='test', args=tuple())
    proc.daemon = True
    proc.start()

    return True


if __name__ == '__main__':
    uvicorn.run("router:app", port=8484, log_level="info")
