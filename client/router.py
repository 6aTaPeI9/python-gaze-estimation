import json
import cv2
import pickle

from websockets.sync.client import connect


CAM = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 100]


def get_image():
    _, frame = CAM.read()
    _, frame = cv2.imencode('.jpg', frame, ENCODE_PARAMS)
    # data = zlib.compress(pickle.dumps(frame, 0))

    return pickle.dumps(frame, 0)


with connect("ws://127.0.0.1:8486/") as websocket:
    while True:
        img = get_image()
        websocket.send(img)

        frame_data = websocket.recv()
        frame_data = json.loads(frame_data.decode())
        frame=pickle.loads(frame_data.get('frame').encode('latin-1'), fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        print(frame_data.get('coords'))
        cv2.imshow('res_image', frame)
        cv2.waitKey(10)

CAM.release()
cv2.destroyAllWindows()

# res = requests.get('http://127.0.0.1:8486/ping')
# print(res.text)