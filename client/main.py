import cv2
import socket
import struct
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
import traceback
import keyboard

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
fig.show()
PRESSED = False
APRESSED = False
POINTS_LIST = []
AVG_LIST = []
CUR_AVG = None

# CAM = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 100]

def get_conect():
    while True:
        print('Пытаемся подключится')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(30)
        time.sleep(2)

        if client_socket.connect_ex(('127.0.0.1', 8485)) == 0:
            connection = client_socket.makefile('wb')
            return client_socket


def get_image():
    _, frame = CAM.read()
    _, frame = cv2.imencode('.jpg', frame, ENCODE_PARAMS)
    # data = zlib.compress(pickle.dumps(frame, 0))

    return pickle.dumps(frame, 0)


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

    frame_data = json.loads(frame_data.decode())
    frame=pickle.loads(frame_data.get('frame').encode('latin-1'), fix_imports=True, encoding="bytes")

    return frame, frame_data


def check_intersection(p0, p1):
    """
        ray
    """
    if len(AVG_LIST) < 4:
        print('Не достаточно')
        return

    p0 = np.array([p0[0][0], p0[1][0], p0[2][0]])
    p1 = p0 + (p1 - p0) * 0.5

    if ray_intersect_triangle(p0, p1, np.array([AVG_LIST[0], AVG_LIST[1], AVG_LIST[2]])):
        print('Смотрим')
        return True

    if ray_intersect_triangle(p0, p1, np.array([AVG_LIST[0], AVG_LIST[2], AVG_LIST[3]])):
        print('Смотрим')
        return True

    print('Не смотрим')
    return False

def key_press():
    print('нажатие')
    global PRESSED
    PRESSED = True

def key_press_a():
    print('нажали a')
    global APRESSED
    APRESSED = not APRESSED

def key_press_s():
    global POINTS_LIST, AVG_LIST
    POINTS_LIST = []
    AVG_LIST.append(np.array(CUR_AVG))

    if len(AVG_LIST) == 4:
        max_z = max([i[2] for i in AVG_LIST])
        left_x = min([AVG_LIST[0][0], AVG_LIST[3][0]]) - 5
        top_y = min([AVG_LIST[0][1], AVG_LIST[1][1]]) - 5
        right_x = max([AVG_LIST[1][0], AVG_LIST[2][0]]) + 5
        bottom_y = max([AVG_LIST[3][1], AVG_LIST[2][1]]) + 5
        top_l = (left_x, top_y, max_z)
        top_r = (right_x, top_y, max_z)
        bot_r = (right_x, bottom_y, max_z)
        bot_l = (left_x, bottom_y, max_z)
        AVG_LIST = [top_l, top_r, bot_r, bot_l]


def draw_points(points):
    """
        Рисуем точки
    """
    def draw_spehere(point, radius):
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x = radius * np.cos(u) * np.sin(v)
        y = radius * np.sin(u) * np.sin(v)
        z = radius * np.cos(v)
        ax.plot_surface(point[0] + x, point[1] + y, -(point[2] + z))

    global PRESSED, APRESSED, CUR_AVG
    ax.cla()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(-100, 100)

    if not points:
        plt.draw()
        plt.pause(1)
        return

    # coords = pickle.loads(points.get('face').encode('latin-1'))
    # ax.scatter(coords[:, 0], coords[:, 1], -coords[:, 2], c='#e377c2',
    #        cmap="PuBuGn_r")
    # draw_spehere(l_pupil, 5)

    # p1 = pickle.loads(points.get('pwc1').encode('latin-1'))
    # ax.scatter(p1[0], p1[1], p1[2], c='#e377c2', linewidths=10)
    # g1 = pickle.loads(points.get('gaze1').encode('latin-1'))

    # p1 = pickle.loads(points.get('pwc2').encode('latin-1'))
    # ax.scatter(p1[0], p1[1], p1[2], c='#e377c2', linewidths=10)
    # g1 = pickle.loads(points.get('gaze2').encode('latin-1'))

    p1 = pickle.loads(points.get('pwc1').encode('latin-1'))
    ax.scatter(p1[0], p1[1], p1[2], c='#000000', linewidths=2)
    p2 = pickle.loads(points.get('pwc2').encode('latin-1'))
    ax.scatter(p2[0], p2[1], p2[2], c='#000000', linewidths=2)

    g1 = pickle.loads(points.get('gaze1').encode('latin-1'))
    g2 = pickle.loads(points.get('gaze2').encode('latin-1'))
    g3 = (
        (g1[0][0] + g2[0][0]) / 2,
        (g1[1][0] + g2[1][0]) / 2,
        (g1[2][0] + g2[2][0]) / 2
    )

    for p in POINTS_LIST + AVG_LIST:
        ax.scatter(p[0], p[1], p[2], c='#8BC34A', linewidths=2)

    if POINTS_LIST:
        x_avg = sum([i[0] for i in POINTS_LIST]) // len(POINTS_LIST)
        y_avg = sum([i[1] for i in POINTS_LIST]) // len(POINTS_LIST)
        z_avg = sum([i[2] for i in POINTS_LIST]) // len(POINTS_LIST)
        CUR_AVG = (x_avg, y_avg, z_avg)
        ax.scatter(x_avg, y_avg, z_avg, c='#0400ff', linewidths=5)

    ax.plot([p1[0], g3[0]], [p1[1], g3[1]], [p1[2], g3[2]], color='#8BC34A')
    ax.plot([p2[0], g3[0]], [p2[1], g3[1]], [p2[2], g3[2]], color='#8BC34A')

    ax.plot([p1[0], g1[0][0]], [p1[1], g1[1][0]], [p1[2], g1[2][0]], color='#822828')
    ax.plot([p2[0], g2[0][0]], [p2[1], g2[1][0]], [p2[2], g2[2][0]], color='#822828')
    check_intersection(p1, np.array([g3[0], g3[1], g3[2]]))
    # check_intersection(p1, np.array([g1[0][0], g1[1][0], g1[2][0]]))
    # check_intersection(p1, np.array([g2[0][0], g2[1][0], g2[2][0]]))

    if PRESSED:
        PRESSED = False
        POINTS_LIST.append(g3)
        # POINTS_LIST.append((g1[0][0], g1[1][0], g1[2][0]))
        # POINTS_LIST.append((g2[0][0], g2[1][0], g2[2][0]))

    # gaze_d1 = pickle.loads(points.get('gaze_d1').encode('latin-1'))
    # ax.scatter(l_pupil[0], l_pupil[1], l_pupil[2], c='#e377c2', linewidths=10)
    # ax.plot([l_pupil[0], gaze_d1[0][0]], [l_pupil[1], gaze_d1[1][0]], [0, 0], color='g')
    # ax.plot([l_pupil2[0], gaze_d2[0]], [l_pupil2[1], gaze_d2[1]], [l_pupil2[2], gaze_d2[2]])
    # print('l_pupil', l_pupil)
    # print('gaze_d1', gaze_d1)

    plt.draw()
    if APRESSED:
        plt.pause(5)
    plt.pause(0.1)

# conn = get_conect()

# keyboard.add_hotkey('space', key_press)
# keyboard.add_hotkey('a', key_press_a)
# keyboard.add_hotkey('d', key_press_s)

# while True:
#     try:
#         send_image(conn)
#         # print('Отправили')
#         frame, fr_data = recv_image(conn)
#         # print('Получили')
#         frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
#         draw_points(fr_data.get('coords'))
#         cv2.imshow('WebCam', frame)
#         cv2.waitKey(10)
#     except Exception as ex:
#         print(traceback.format_exc())
#         conn = get_conect()

CAM.release()
cv2.destroyAllWindows()

# [[40.4484254 ]
#  [44.49410401]
#  [95.13622113]]

# [[-130.46700186]
#  [ -14.15731318]
#  [ 486.45219612]]

# [[41.92203925]
#  [19.73298953]
#  [95.89995664]]