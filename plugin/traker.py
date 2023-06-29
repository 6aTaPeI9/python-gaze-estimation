import tools
import numpy as np
import json


CALIB_MAP = {
    'botLeft': None,
    'botRight': None,
    'topRight': None,
    'topLeft': None,
}


def monitor_calibrate(calib_map: dict):
    """
        Калибровка углов монитора
    """
    print('Калибровка по: ')
    left_eye_avg = []
    right_eye_avg = []

    # Вычисляем средние координаты для левого глаза
    # Идем по каждому ключу key_m и для списка координат\
    for k in ['topLeft', 'topRight', 'botRight', 'botLeft']:
        coords = calib_map.get(k)
        x_avg = sum([i[2][0] for i in coords]) // len(coords)
        y_avg = sum([i[2][1] for i in coords]) // len(coords)
        z_avg = sum([i[2][2] for i in coords]) // len(coords)
        cur_avg = (x_avg, y_avg, z_avg)
        left_eye_avg.append(cur_avg)

    # Вычисляем средние координаты для право глаза
    for k in ['topLeft', 'topRight', 'botRight', 'botLeft']:
        coords = calib_map.get(k)
        x_avg = sum([i[3][0] for i in coords]) // len(coords)
        y_avg = sum([i[3][1] for i in coords]) // len(coords)
        z_avg = sum([i[3][2] for i in coords]) // len(coords)
        cur_avg = (x_avg, y_avg, z_avg)
        right_eye_avg.append(cur_avg)

    max_z = max([i[2] for i in left_eye_avg])
    left_x = min([left_eye_avg[0][0], left_eye_avg[3][0]]) - 10
    top_y = min([left_eye_avg[0][1], left_eye_avg[1][1]]) - 10
    right_x = max([left_eye_avg[1][0], left_eye_avg[2][0]]) + 10
    bottom_y = max([left_eye_avg[3][1], left_eye_avg[2][1]]) + 10
    top_l = (left_x, top_y, max_z)
    top_r = (right_x, top_y, max_z)
    bot_r = (right_x, bottom_y, max_z)
    bot_l = (left_x, bottom_y, max_z)
    left_eye_avg = [top_l, top_r, bot_r, bot_l]

    max_z = max([i[2] for i in right_eye_avg])
    left_x = min([right_eye_avg[0][0], right_eye_avg[3][0]]) - 10
    top_y = min([right_eye_avg[0][1], right_eye_avg[1][1]]) - 10
    right_x = max([right_eye_avg[1][0], right_eye_avg[2][0]]) + 10
    bottom_y = max([right_eye_avg[3][1], right_eye_avg[2][1]]) + 10
    top_l = (left_x, top_y, max_z)
    top_r = (right_x, top_y, max_z)
    bot_r = (right_x, bottom_y, max_z)
    bot_l = (left_x, bottom_y, max_z)
    right_eye_avg = [top_l, top_r, bot_r, bot_l]
    print('right_eye_avg:', right_eye_avg)
    print('left_eye_avg:', left_eye_avg)

    return left_eye_avg, right_eye_avg


def ray_intersect(p1, p2, g1, g2, left_mon, right_mon):
    p1 = np.array(p1)
    p2 = np.array(p2)
    g1 = np.array(g1)
    g2 = np.array(g2)

    g1 = p1 + (g1 - p1) * 0.5
    g2 = p2 + (g2 - p2) * 0.5

    if tools.ray_intersect_triangle(p1, g1, np.array([left_mon[0], left_mon[1], left_mon[2]])):
        return 'watch_left'

    if tools.ray_intersect_triangle(p1, g1, np.array([left_mon[0], left_mon[2], left_mon[3]])):
        return 'watch_left'

    if tools.ray_intersect_triangle(p2, g2, np.array([right_mon[0], right_mon[1], right_mon[2]])):
        return 'watch_right'

    if tools.ray_intersect_triangle(p2, g2, np.array([right_mon[0], right_mon[2], right_mon[3]])):
        return 'watch_right'

    return 'do_not_watch'
