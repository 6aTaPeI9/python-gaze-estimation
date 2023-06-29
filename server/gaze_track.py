import mediapipe as mp
import cv2
import traceback

import json
import pickle
import struct
import base64
import numpy as np

from gaze import gazestimation


def calc_gaze(frame_data: str, w_frame=True):
    """
        Вычисление направления взгляда
    """
    try:
        frame_data = base64.b64decode(frame_data)
        frame_data = np.frombuffer(frame_data, dtype=np.uint8)
        image = cv2.imdecode(frame_data, flags=1)
        # frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        # image = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        # ---------------------------------------------------------
        gz = gazestimation.Gazetimation()
        image, coords = gz.run(frame=image)
        # ---------------------------------------------------------

        _, image = cv2.imencode('.jpg', image)
        fr_res = base64.b64encode(image)

        data = {}
        if w_frame:
            data['frame'] = fr_res.decode()

        data.update(coords)

        return data
    except Exception:
        print(traceback.format_exc())
        return data