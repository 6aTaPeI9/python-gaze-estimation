import mediapipe as mp
import cv2
import traceback

import json
import pickle
import struct
import base64
import numpy as np

from gaze import gazestimation


IMAGE_ENCODE = [int(cv2.IMWRITE_JPEG_QUALITY), 100]


def calc_gaze(frame_data: bytes):
    """
        Вычисление направления взгляда
    """
    try:
        if isinstance(frame_data, str):
            frame_data = base64.b64decode(frame_data)
            frame_data = np.frombuffer(frame_data, dtype=np.uint8)
            image = cv2.imdecode(frame_data, flags=1)
        else:
            frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            image = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        # ---------------------------------------------------------
        gz = gazestimation.Gazetimation()
        image, coords = gz.run(frame=image)
        # ---------------------------------------------------------

        result, image = cv2.imencode('.jpg', image)
        data = base64.b64encode(image)
        data = json.dumps({
            'frame': data.decode(),
            # 'coords': coords
        })

        return data
    except Exception as ex:
        print(traceback.format_exc())
    pass