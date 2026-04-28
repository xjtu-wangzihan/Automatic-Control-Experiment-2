# qlabs_setup_pro.py
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.basic_shape import QLabsBasicShape
import pal.resources.rtmodels as rtmodels
import time
from qvl.spline_line import QLabsSplineLine
def setup():
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("无法连接 QLabs，请检查软件是否已打开")
        return
    
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(1)
    
    # 生成小车
    hqcar = QLabsQCar2(qlabs)
    hqcar.spawn_id(actorNumber=0, location=[0, 0, 0.005], rotation=[0,0,0], waitForConfirmation=True)
    
    line_length = 400.0  # 线长400米
    lineWidth = 0.1
    lineHeight = 0.01
    sYellow = QLabsSplineLine(qlabs)
    sYellow.spawn(location=[0, 0, 0], scale=[1,1,1], configuration=1)
    sYellow.set_points(color=[1, 1, 0], pointList=[[0, 0, lineHeight, lineWidth], 
                                                   [line_length, 0, lineHeight, lineWidth]])

    # 2. 绿色车道线 (y = -2)
    sGreen = QLabsSplineLine(qlabs)
    sGreen.spawn(location=[0, 0, 0], scale=[1,1,1], configuration=1)
    sGreen.set_points(color=[0, 1, 0], pointList=[[0, -2, lineHeight, lineWidth], 
                                                  [line_length, -2, lineHeight, lineWidth]])
    
    # 生成直线上的红色障碍物
    for i in range(6):
            shape = QLabsBasicShape(qlabs)
            # 车道 Y 坐标：左车道 0.75, 右车道 -0.75
            lane_y = 0 if i % 2 == 0 else -2
            pos_x = 30.0 + i * 30.0  # 每隔30米一个障碍物
            
            shape.spawn_id(actorNumber=i+10, 
                        location=[pos_x, lane_y, 0.25], 
                        scale=[0.4, 0.4, 0.6], 
                        configuration=shape.SHAPE_CUBE, 
                        waitForConfirmation=True)
            shape.set_material_properties(color=[1, 0, 0]) # 亮红色

    # 设置视角
    hcamera = QLabsFreeCamera(qlabs)
    hcamera.spawn([-3.0, 0.0, 2.0], [0, 0.2, 0])
    hqcar.possess()
    
    # 启动实时模型
    QLabsRealTime().start_real_time_model(rtmodels.QCAR2)
    print("--- 场景已就绪，QLabs 应显示 Running 状态 ---")

if __name__ == '__main__':
    setup()