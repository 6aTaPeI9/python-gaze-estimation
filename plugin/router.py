"""
    Роутинг запросов
"""
import uvicorn
from pydantic import BaseModel

import time
import json
import sys
import os
import multiprocessing
import configparser
import traceback
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
CUR_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
CONFIG = configparser.ConfigParser()
CONFIG.read(CUR_DIR + '\\config.conf')
os.makedirs(FL_DIR_PATH, exist_ok=True)

app.mount("/calibrate", StaticFiles(directory=CUR_DIR + '\\calibration', html = True), name="static")
DEV_UUID = uuid.uuid4()
PROC_RUNNED = False

def sub_process():
    def get_image():
        _, frame = CAM.read()
        _, image = cv2.imencode('.jpg', frame)
        image = base64.b64encode(image).decode()
        return {'frame': image}

    import cv2
    import base64
    import requests

    CAM = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with open(FL_PATH, 'r+') as fl:
        dt = fl.readline()
    dt = json.loads(dt)

    while True:
        try:
            time.sleep(1)
            with connect(f'ws://{CONFIG["gaze_server"]["host"]}/') as websocket:
                time.sleep(1)
                for i in range(10):
                    time.sleep(1)
                    try:
                        requests.get(f'http://{CONFIG["local_tracker"]["host"]}/ping/')
                    except Exception as ex:
                        break

                    try:
                        websocket.send(json.dumps(get_image()))
                        res = websocket.recv()
                        res = json.loads(res)

                        if res.get('pwc1'):
                            status = ray_intersect(res['pwc1'], res['pwc2'], res['gaze1'], res['gaze2'], dt['left_mon'], dt['right_mon'])
                        else:
                            status = 'do_not_watch'

                        with open(FL_PATH, 'w+') as fl:
                            fl.write(json.dumps({
                                'face': res.get('frame'),
                                'status': status,
                                **dt
                            }))

                        adm_dt = {
                            'device': str(DEV_UUID),
                            'photo': res.get('frame'),
                            'status': status
                        }
                        # Сохраняем полученное в админку
                        requests.post(f'http://{CONFIG["admin_panel"]["host"]}/save_process/', json.dumps(adm_dt))
                    except Exception as ex:
                        if '-215:Assertion failed' in str(ex):
                            CAM = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                        print(traceback.format_exc())
        except Exception as ex:
            print(traceback.format_exc())

@app.post("/status")
def status():
    """
        API для получения текущего статуса устройства.
        Отправляет данные:
            pwc1, pwc2, gaze1, gaze2
            frame: последнее обработанное изображение
            track_runned: запущен ли процесс отслеживания
            status: do_not_watch/watch_left/watch_right
    """
    global PROC_RUNNED
    with open(FL_PATH, 'r+') as fl:
        dt = fl.readline()

    dt = json.loads(dt or '{}')
    dt['track_runned'] = PROC_RUNNED
    return dt

@app.get("/ping")
def pong():
    """
        Тест эхо
    """
    return {"ping": "pong!"}

@app.get("/run_calib")
def run_calib():
    """
        Тест эхо
    """
    global PROC_RUNNED
    if not PROC_RUNNED:
        PROC_RUNNED = True
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

    global PROC_RUNNED
    if not PROC_RUNNED:
        PROC_RUNNED = True
        proc = multiprocessing.Process(target=sub_process, name='test', args=tuple())
        proc.daemon = True
        proc.start()

    return True


if __name__ == '__main__':
    uvicorn.run(app, port=8484, log_level="info")