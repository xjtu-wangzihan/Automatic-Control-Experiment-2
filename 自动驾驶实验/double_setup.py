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

def setup():
    os.system('cls')
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if (not qlabs.open("localhost")):
        print("Unable to connect to QLabs")
        sys.exit()
        return
    print("Connected to QLabs")
    
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(1)
    
    # 双车道参数（沿 x 方向前进，y 方向为横向偏移）
    y_min = -5.0
    y_max = 5.0
    lane_width = (y_max - y_min) / 2.0
    # 仿真坐标系下：y 较大为左车道，y 较小为右车道
    left_lane_center = y_min + 1.5 * lane_width
    right_lane_center = y_min + 0.5 * lane_width

    # 生成小车
    hqcar = QLabsQCar2(qlabs)
    hqcar.spawn_id(
        actorNumber=0,
        location=[0, left_lane_center, 0.1],
        rotation=[0, 0, 0],
        waitForConfirmation=True
    )

    # 可视化两车道边界线和中分线（y=-5, y=0, y=5）
    z_line = 0.02
    lane_line_width = 0.18
    x_start = -5.0
    x_end = 180.0

    left_boundary = QLabsSplineLine(qlabs)
    left_boundary.spawn(location=[0, 0, 0], scale=[1, 1, 1], configuration=1)
    left_boundary.set_points(
        color=[1, 1, 1],
        pointList=[[x_start, y_min, z_line, lane_line_width], [x_end, y_min, z_line, lane_line_width]],
        alignEndPointTangents=False,
    )

    # 中间分界线改为白色虚线
    dash_len = 4.0
    gap_len = 4.0
    x_dash = x_start
    while x_dash < x_end:
        x_dash_end = min(x_dash + dash_len, x_end)
        lane_dash = QLabsSplineLine(qlabs)
        lane_dash.spawn(location=[0, 0, 0], scale=[1, 1, 1], configuration=1)
        lane_dash.set_points(
            color=[1, 1, 1],
            pointList=[[x_dash, 0.0, z_line, lane_line_width], [x_dash_end, 0.0, z_line, lane_line_width]],
            alignEndPointTangents=False,
        )
        x_dash += dash_len + gap_len

    right_boundary = QLabsSplineLine(qlabs)
    right_boundary.spawn(location=[0, 0, 0], scale=[1, 1, 1], configuration=1)
    right_boundary.set_points(
        color=[1, 1, 1],
        pointList=[[x_start, y_max, z_line, lane_line_width], [x_end, y_max, z_line, lane_line_width]],
        alignEndPointTangents=False,
    )
    
    # 在两条车道交错放置障碍物（共10个）
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
        shape.set_material_properties(color=[1, 0, 0]) # 亮红色

    # 设置视角
    hcamera = QLabsFreeCamera(qlabs)
    hcamera.spawn([-6.0, 0.0, 9.5], [0.0, 0.70, 0.0])
    hqcar.possess()
    
    # 启动实时模型
    QLabsRealTime().start_real_time_model(rtmodels.QCAR2)
    print("--- 场景已就绪：双车道(y=-5到5) + 10个交错障碍物 ---")

if __name__ == '__main__':
    setup()