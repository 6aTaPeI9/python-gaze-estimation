import mediapipe as mp
import cv2
import traceback

import json
import pickle
import struct
import base64
import numpy as np

from gaze import gazestimation


def calc_gaze(frame_data: str):
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
        data = base64.b64encode(image)
        data = json.dumps({
            'frame': data.decode(),
            **coords
        })

        return data
    except Exception as ex:
        print(traceback.format_exc())
    pass