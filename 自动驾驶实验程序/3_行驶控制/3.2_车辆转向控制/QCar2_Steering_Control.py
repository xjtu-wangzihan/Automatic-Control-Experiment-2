# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports

"""
QCar2_Steering_Control.py
本程序用于QCar2小车横向（转向）控制实验，演示如何基于Stanley算法实现路径跟踪与转向控制，并通过MultiScope实时可视化车辆状态。
支持仿真与实车两种模式，包含安全中断、线程控制、路径生成等功能。
"""
import signal
import numpy as np
from threading import Thread
import time
import cv2
import pyqtgraph as pg

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images
from pal.utilities.math import *

#================ 实验参数配置 ================
# ===== 时间参数 =====
# tf: 实验总时长（秒）
# startDelay: 启动延迟，滤波器稳定用（秒）
# controllerUpdateRate: 控制循环频率（Hz），不建议超过500
tf = 20
startDelay = 1
controllerUpdateRate = 100
sampleTime = 1/controllerUpdateRate

# ===== 速度控制器参数 =====
# v_ref: 期望速度（m/s）
# K_p: 比例增益
# K_i: 积分增益
v_ref = 1
K_p = 0.1
K_i = 1

# ===== 转向控制器参数 =====
# enableSteeringControl: 是否启用转向控制
# K_stanley: Stanley控制器增益
# nodeSequence: 路径节点序列，用于路径生成
enableSteeringControl = True
K_stanley = 1
nodeSequence = [9, 14, 9]

# 标定初始位姿
#calibrationPose = [0,0,-np.pi/2]
calibrationPose = [0,2,-np.pi/2]

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 初始设置
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
else:
    initialPose = [0, 0, 0]

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]]
    )
    calibrate = False
else:
    calibrate =  'y' in input('是否需要重新标定？(y/n)')

# 用于安全地通过键盘中断实验
global KILL_THREAD
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)
#endregion


# 速度PI控制器类
class SpeedController:
    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3  # 最大油门限制
        self.kp = kp
        self.ki = ki
        self.ei = 0  # 积分项

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += e * dt
        u = self.kp * e + self.ki * self.ei
        u = np.clip(u, -self.maxThrottle, self.maxThrottle)
        return u


# Stanley转向控制器类
class SteeringController:
    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi/6  # 最大转向角
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic
        self.p_ref = (0, 0)
        self.th_ref = 0

    # Stanley算法更新
    def update(self, p, th, speed):
        wp_1 = self.wp[:, np.mod(self.wpi, self.N-1)]
        wp_2 = self.wp[:, np.mod(self.wpi+1, self.N-1)]
        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except ZeroDivisionError:
            return 0
        tangent = np.arctan2(v_uv[1], v_uv[0])
        s = np.dot(p-wp_1, v_uv)
        if s >= v_mag:
            if  self.cyclic or self.wpi < self.N-2:
                self.wpi += 1
        ep = wp_1 + v_uv*s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)
        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent-th)
        self.p_ref = ep
        self.th_ref = tangent
        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k*ect, speed)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle)

# 控制主循环，负责读取传感器、EKF状态估计、PI速度控制、Stanley转向控制、写入指令、刷新可视化
def controlLoop():
    #region : 控制循环初始化
    global KILL_THREAD
    u = 0
    delta = 0
    # 用于将数据采样频率限制为10Hz
    countMax = controllerUpdateRate / 10
    count = 0
    #endregion

    #region : 控制器初始化
    speedController = SpeedController(
        kp=K_p,
        ki=K_i
    )
    if enableSteeringControl:
        steeringController = SteeringController(
            waypoints=waypointSequence,
            k=K_stanley
        )
    #endregion

    #region : QCar接口初始化
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl or calibrate:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)
    else:
        gps = memoryview(b'')
    #endregion

    with qcar, gps:
        t0 = time.time()
        t = 0
        while (t < tf + startDelay) and (not KILL_THREAD):
            #region : 循环计时
            tp = t
            t = time.time() - t0
            dt = t - tp
            #endregion

            #region : 读取传感器并状态估计
            qcar.read()
            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array([
                        gps.position[0],
                        gps.position[1],
                        gps.orientation[2]
                    ])
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        y_gps,
                        qcar.gyroscope[2],
                    )
                else:
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        None,
                        qcar.gyroscope[2],
                    )
                x = ekf.x_hat[0,0]
                y = ekf.x_hat[1,0]
                th = ekf.x_hat[2,0]
                p = ( np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2)
            v = qcar.motorTach
            #endregion

            #region : 控制器更新与指令写入
            if t < startDelay:
                u = 0
                delta = 0
            else:
                u = speedController.update(v, v_ref, dt)
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v)
                else:
                    delta = 0
            #endregion

            qcar.write(u, delta)

            #region : 刷新可视化曲线
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay
                # 多通道Scope采样
                if enableSteeringControl:
                    steeringScope.axes[4].sample(t_plot, [[p[0], p[1]]])
                    p[0] = ekf.x_hat[0,0]
                    p[1] = ekf.x_hat[1,0]
                    x_ref = steeringController.p_ref[0]
                    y_ref = steeringController.p_ref[1]
                    th_ref = steeringController.th_ref
                    x_ref = gps.position[0]
                    y_ref = gps.position[1]
                    th_ref = gps.orientation[2]
                    steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                    steeringScope.axes[1].sample(t_plot, [p[1], y_ref])
                    steeringScope.axes[2].sample(t_plot, [th, th_ref])
                    steeringScope.axes[3].sample(t_plot, [delta])
                    arrow.setPos(p[0], p[1])
                    arrow.setStyle(angle=180-th*180/np.pi)
                count = 0
            #endregion
            continue
        qcar.read_write_std(throttle=0, steering=0)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 实验主流程
if __name__ == '__main__':


    #region : Scope可视化配置
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30

    # 多通道Scope用于监控转向控制
    if enableSteeringControl:
        steeringScope = MultiScope(
            rows=4,
            cols=2,
            title='车辆侧向控制',
            fps=fps
        )
        steeringScope.addAxis(
            row=0,
            col=0,
            timeWindow=tf,
            yLabel='X方向位置 [m]',
            yLim=(-2.5, 2.5)
        )
        steeringScope.axes[0].attachSignal(name='x测量值')
        steeringScope.axes[0].attachSignal(name='x期望值')
        steeringScope.addAxis(
            row=1,
            col=0,
            timeWindow=tf,
            yLabel='Y方向位置 [m]',
            yLim=(-1, 5)
        )
        steeringScope.axes[1].attachSignal(name='y测量值')
        steeringScope.axes[1].attachSignal(name='y期望值')
        steeringScope.addAxis(
            row=2,
            col=0,
            timeWindow=tf,
            yLabel='朝向 [rad]',
            yLim=(-3.5, 3.5)
        )
        steeringScope.axes[2].attachSignal(name='朝向测量值')
        steeringScope.axes[2].attachSignal(name='朝向期望值')
        steeringScope.addAxis(
            row=3,
            col=0,
            timeWindow=tf,
            yLabel='前轮转角 [rad]',
            yLim=(-0.6, 0.6)
        )
        steeringScope.axes[3].attachSignal()
        steeringScope.axes[3].xLabel = '时间 [s]'
        steeringScope.addXYAxis(
            row=0,
            col=1,
            rowSpan=4,
            xLabel='x 位置 [m]',
            yLabel='y 位置 [m]',
            xLim=(-2.5, 2.5),
            yLim=(-1, 5)
        )
        im = cv2.imread(
            images.SDCS_CITYSCAPE,
            cv2.IMREAD_GRAYSCALE
        )
        steeringScope.axes[4].attachImage(
            scale=(-0.002035, 0.002035),
            offset=(1125,2365),
            rotation=180,
            levels=(0, 255)
        )
        steeringScope.axes[4].images[0].setImage(image=im)
        referencePath = pg.PlotDataItem(
            pen={'color': (85,168,104), 'width': 2},
            name='Reference'
        )
        steeringScope.axes[4].plot.addItem(referencePath)
        referencePath.setData(waypointSequence[0, :],waypointSequence[1, :])
        steeringScope.axes[4].attachSignal(name='Estimated', width=2)
        arrow = pg.ArrowItem(
            angle=180,
            tipAngle=60,
            headLen=10,
            tailLen=10,
            tailWidth=5,
            pen={'color': 'w', 'fillColor': [196,78,82], 'width': 1},
            brush=[196,78,82]
        )
        arrow.setPos(initialPose[0], initialPose[1])
        steeringScope.axes[4].plot.addItem(arrow)
    #endregion

    #region : 启动控制线程并运行实验
    controlThread = Thread(target=controlLoop)
    controlThread.start()
    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
    #endregion
    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()
    input('实验结束，按任意键退出...')
#endregion