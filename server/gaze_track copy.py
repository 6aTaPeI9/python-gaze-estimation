import mediapipe as mp
import cv2

import json
import socket
import pickle
import struct
import fastapi
import uvicorn

from gaze import gazestimation



HOST=''
PORT=8485
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((HOST,PORT))
s.listen(10)
print('Ждем коннект')
conn, addr = s.accept()
data = b""
payload_size = struct.calcsize(">L")
IMAGE_ENCODE = [int(cv2.IMWRITE_JPEG_QUALITY), 100]

mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model
print('STARTED')

def recv_image(conn):
    data = b''
    payload_size = struct.calcsize(">L")

    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")

    return frame


while True:
    try:
        frame = recv_image(conn)
        image = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        # ---------------------------------------------------------
        gz = gazestimation.Gazetimation()
        image, coords = gz.run(frame=image)
        # ---------------------------------------------------------

        result, frame = cv2.imencode('.jpg', image, encode_param)
        data = pickle.dumps(frame, 0).decode('latin-1')
        data = json.dumps({
            'frame': data,
            'coords': coords
        }).encode()
        size = len(data)

        conn.sendall(struct.pack(">L", size) + data)
        data = b""
        # cv2.imshow('output window', image)
    except Exception as ex:
        print(ex)

cap.release()


def calc_gaze(image: bytes):
    """
        Вычисление направления взгляда
    """
    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    image = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    gz = gazestimation.Gazetimation()
    image, coords = gz.run(frame=image)
    pass