import tools
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import keyboard

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
fig.show()

CALIB_QUEUE = [
    'left_bottom',
    'right_bottom',
    'right_top',
    'left_top',
]
CALIB_MAP = {}
CUR_POINT = []

SAVE_KEY = False
GO_NEXT_KEY = False

LEFT_EYE_MONITOR = []
RIGHT_EYE_MONITOR = []

def key_save_point():
    global SAVE_KEY
    SAVE_KEY = True
    pass


def key_go_next():
    global GO_NEXT_KEY
    GO_NEXT_KEY = True
    pass


def togle_next_key(calib_map: dict):
    global CUR_POINT, SAVE_KEY, GO_NEXT_KEY

    if not CALIB_QUEUE:
        print('Калибруем')
        CUR_POINT = []
        GO_NEXT_KEY = False
        monitor_calibrate(calib_map)
        return

    cur_key = CALIB_QUEUE.pop()
    print('Сохраняем для: ', cur_key)
    calib_map[cur_key] = CUR_POINT
    CUR_POINT = []
    SAVE_KEY = False
    GO_NEXT_KEY = False


def median_coord(calib_map: dict):
    """
        Вычисление средней точки из массива
    """
    pass


def monitor_calibrate(calib_map: dict):
    """
        Калибровка углов монитора
    """
    print('Калибровка по: ')
    left_eye_avg = []
    right_eye_avg = []

    for key_m, coords in calib_map.items():
        x_avg = sum([i[2][0] for i in coords]) // len(coords)
        y_avg = sum([i[2][1] for i in coords]) // len(coords)
        z_avg = sum([i[2][2] for i in coords]) // len(coords)
        cur_avg = (x_avg, y_avg, z_avg)
        left_eye_avg.append(cur_avg)

    for key_m, coords in calib_map.items():
        x_avg = sum([i[3][0] for i in coords]) // len(coords)
        y_avg = sum([i[3][1] for i in coords]) // len(coords)
        z_avg = sum([i[3][2] for i in coords]) // len(coords)
        cur_avg = (x_avg, y_avg, z_avg)
        right_eye_avg.append(cur_avg)

    max_z = max([i[2] for i in left_eye_avg])
    left_x = min([left_eye_avg[0][0], left_eye_avg[3][0]]) - 5
    top_y = min([left_eye_avg[0][1], left_eye_avg[1][1]]) - 5
    right_x = max([left_eye_avg[1][0], left_eye_avg[2][0]]) + 5
    bottom_y = max([left_eye_avg[3][1], left_eye_avg[2][1]]) + 5
    top_l = (left_x, top_y, max_z)
    top_r = (right_x, top_y, max_z)
    bot_r = (right_x, bottom_y, max_z)
    bot_l = (left_x, bottom_y, max_z)
    left_eye_avg = [top_l, top_r, bot_r, bot_l]

    max_z = max([i[2] for i in right_eye_avg])
    left_x = min([right_eye_avg[0][0], right_eye_avg[3][0]]) - 5
    top_y = min([right_eye_avg[0][1], right_eye_avg[1][1]]) - 5
    right_x = max([right_eye_avg[1][0], right_eye_avg[2][0]]) + 5
    bottom_y = max([right_eye_avg[3][1], right_eye_avg[2][1]]) + 5
    top_l = (left_x, top_y, max_z)
    top_r = (right_x, top_y, max_z)
    bot_r = (right_x, bottom_y, max_z)
    bot_l = (left_x, bottom_y, max_z)
    right_eye_avg = [top_l, top_r, bot_r, bot_l]
    RIGHT_EYE_MONITOR.extend(right_eye_avg)
    LEFT_EYE_MONITOR.extend(left_eye_avg)

    print('right_eye_avg:', right_eye_avg)
    print('left_eye_avg:', left_eye_avg)


def ray_intersect(p1, p2, g1, g2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    g1 = np.array(g1)
    g2 = np.array(g2)

    g1 = p1 + (g1 - p1) * 0.5
    g2 = p2 + (g2 - p2) * 0.5

    if tools.ray_intersect_triangle(p1, g1, np.array([LEFT_EYE_MONITOR[0], LEFT_EYE_MONITOR[1], LEFT_EYE_MONITOR[2]])):
        print('ЛЕВЫЙ ГЛАЗ: Смотрим')
        return True

    if tools.ray_intersect_triangle(p1, g1, np.array([LEFT_EYE_MONITOR[0], LEFT_EYE_MONITOR[2], LEFT_EYE_MONITOR[3]])):
        print('ЛЕВЫЙ ГЛАЗ: Смотрим')
        return True

    if tools.ray_intersect_triangle(p2, g2, np.array([RIGHT_EYE_MONITOR[0], RIGHT_EYE_MONITOR[1], RIGHT_EYE_MONITOR[2]])):
        print('ПРАВЫЙ ГЛАЗ: Смотрим')
        return True

    if tools.ray_intersect_triangle(p2, g2, np.array([RIGHT_EYE_MONITOR[0], RIGHT_EYE_MONITOR[2], RIGHT_EYE_MONITOR[3]])):
        print('ПРАВЫЙ ГЛАЗ: Смотрим')
        return True

    print('ВООБЩЕ НЕ СМОТРИМ')
    return False


def draw_matplot(pwc1: list, pwc2: list, gaze1: list, gaze2: list):
    """
        Рендер матплота
    """
    global SAVE_KEY
    def draw_spehere(point, radius):
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x = radius * np.cos(u) * np.sin(v)
        y = radius * np.sin(u) * np.sin(v)
        z = radius * np.cos(v)
        ax.plot_surface(point[0] + x, point[1] + y, -(point[2] + z))

    if SAVE_KEY and CALIB_QUEUE:
        if len(CUR_POINT) < 4:
            CUR_POINT.append([pwc1, pwc2, gaze1, gaze2])
        else:
            print('Закончили')
            SAVE_KEY = False

    if GO_NEXT_KEY:
        togle_next_key(CALIB_MAP)

    ax.cla()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(-100, 100)

    ax.scatter(*pwc1, c='#000000', linewidths=2)
    ax.scatter(*pwc2, c='#000000', linewidths=2)
    ax.scatter(*gaze1, c='#000000', linewidths=2)
    ax.scatter(*gaze2, c='#000000', linewidths=2)

    for coord in LEFT_EYE_MONITOR:
        ax.scatter(*coord, c='#20B2AA', linewidths=3)

    for coord in RIGHT_EYE_MONITOR:
        ax.scatter(*coord, c='#00FF00', linewidths=3)

    if LEFT_EYE_MONITOR and RIGHT_EYE_MONITOR:
        ray_intersect(pwc1, pwc2, gaze1, gaze2)

    g3 = (
        (gaze1[0] + gaze2[0]) / 2,
        (gaze1[1] + gaze2[1]) / 2,
        (gaze1[2] + gaze2[2]) / 2
    )

    plt.draw()
    plt.pause(0.1)

keyboard.add_hotkey('space', key_save_point)
keyboard.add_hotkey('a', key_go_next)