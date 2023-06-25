import json
import cv2
import base64
import numpy as np
import traker

from websockets.sync.client import connect

ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
CAM = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def get_image():
    _, frame = CAM.read()
    _, image = cv2.imencode('.jpg', frame)
    image = base64.b64encode(image).decode()
    return image


def process_img(websocket):
    """
        Обработка изображения
    """
    img = get_image()
    websocket.send(img)

    frame_data = websocket.recv()
    frame_data = json.loads(frame_data)
    frame = base64.b64decode(frame_data.get('frame'))
    frame = np.frombuffer(frame, dtype=np.uint8)
    image = cv2.imdecode(frame, flags=1)
    return image, frame_data

with connect("ws://127.0.0.1:8485/") as websocket:
    while True:
        image, frame_data = process_img(websocket)
        traker.draw_matplot(
            frame_data['pwc1'],
            frame_data['pwc2'],
            frame_data['gaze1'],
            frame_data['gaze2']
        )
        cv2.imshow('res_image', image)
        cv2.waitKey(10)
        # frame=pickle.loads(frame_data.get('frame').encode('latin-1'), fix_imports=True, encoding="bytes")
        # frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        # websocket.close()
        # break

CAM.release()
cv2.destroyAllWindows()

# res = requests.get('http://127.0.0.1:8486/ping')
# print(res.text)