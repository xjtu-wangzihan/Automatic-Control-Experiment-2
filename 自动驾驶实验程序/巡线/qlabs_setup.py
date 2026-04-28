from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.basic_shape import QLabsBasicShape
from qvl.spline_line import QLabsSplineLine
import pal.resources.rtmodels as rtmodels
import time
import os
import sys


def spawn_straight_line(qlabs, x_start, x_end, y_pos, z_line, line_width, color):
    line = QLabsSplineLine(qlabs)
    line.spawn(location=[0, 0, 0], scale=[1, 1, 1], configuration=1)
    line.set_points(
        color=color,
        pointList=[[x_start, y_pos, z_line, line_width], [x_end, y_pos, z_line, line_width]],
        alignEndPointTangents=False,
    )


def setup():
    os.system('cls')
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if not qlabs.open("localhost"):
        print("Unable to connect to QLabs")
        sys.exit()
        return
    print("Connected to QLabs")

    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(1)

    y_min = -5.0
    y_max = 5.0
    lane_width = (y_max - y_min) / 2.0
    left_lane_center = y_min + 1.5 * lane_width
    right_lane_center = y_min + 0.5 * lane_width

    hqcar = QLabsQCar2(qlabs)
    hqcar.spawn_id(
        actorNumber=0,
        location=[0, left_lane_center, 0.1],
        rotation=[0, 0, 0],
        waitForConfirmation=True,
    )

    z_line = 0.02
    lane_line_width = 0.18
    guide_line_width = 0.12
    x_start = -5.0
    x_end = 180.0

    spawn_straight_line(
        qlabs,
        x_start=x_start,
        x_end=x_end,
        y_pos=y_min,
        z_line=z_line,
        line_width=lane_line_width,
        color=[1, 1, 1],
    )
    spawn_straight_line(
        qlabs,
        x_start=x_start,
        x_end=x_end,
        y_pos=y_max,
        z_line=z_line,
        line_width=lane_line_width,
        color=[1, 1, 1],
    )

    dash_len = 4.0
    gap_len = 4.0
    x_dash = x_start
    while x_dash < x_end:
        dash_end = min(x_dash + dash_len, x_end)
        spawn_straight_line(
            qlabs,
            x_start=x_dash,
            x_end=dash_end,
            y_pos=0.0,
            z_line=z_line,
            line_width=lane_line_width,
            color=[1, 1, 1],
        )
        x_dash += dash_len + gap_len

    # 在每条车道中间放置一条显式引导线，供主程序按颜色直接循迹。
    spawn_straight_line(
        qlabs,
        x_start=x_start,
        x_end=x_end,
        y_pos=left_lane_center,
        z_line=z_line + 0.001,
        line_width=guide_line_width,
        color=[1, 1, 0],
    )
    spawn_straight_line(
        qlabs,
        x_start=x_start,
        x_end=x_end,
        y_pos=right_lane_center,
        z_line=z_line + 0.001,
        line_width=guide_line_width,
        color=[0, 1, 0],
    )

    for i in range(10):
        shape = QLabsBasicShape(qlabs)
        pos_x = 18.0 + i * 18.0
        pos_y = left_lane_center if (i % 2 == 0) else right_lane_center
        shape.spawn_id(
            actorNumber=i + 10,
            location=[pos_x, pos_y, 0.25],
            scale=[0.45, 0.45, 0.6],
            configuration=shape.SHAPE_CUBE,
            waitForConfirmation=True,
        )
        shape.set_material_properties(color=[1, 0, 0])

    hcamera = QLabsFreeCamera(qlabs)
    hcamera.spawn([-6.0, 0.0, 9.5], [0.0, 0.70, 0.0])
    hqcar.possess()

    QLabsRealTime().start_real_time_model(rtmodels.QCAR2)
    print("--- 场景已就绪：白色双车道边界 + 左黄右绿中心引导线 + 10个交错障碍物 ---")


if __name__ == '__main__':
    setup()
